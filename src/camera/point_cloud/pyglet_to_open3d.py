# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 RealSense, Inc. All Rights Reserved.

"""
RealSense 深度点云 + Open3D 可视化（原 pyglet 演示逻辑，已去掉 pyglet）。

彩色预览窗口 color_image 聚焦时按键：
    [p]     暂停 / 继续
    [r]     重置选项（降采样档位与后处理关闭）
    [d]     循环降采样档位 (0→1→2)
    [c]     切换彩色纹理 / 深度伪彩色纹理
    [f]     开关深度后处理链
    [s]     保存当前彩色帧为 ./out.png
    [t]     保存调试包到 test_data/realsense_open3d
    [e]     导出当前帧点云 ./out.ply
    [q][ESC] 退出
Open3D 窗口：鼠标旋转缩放点云。
"""

import os
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d


class AppState:

    def __init__(self):
        self.paused = False
        self.decimate = 0
        self.color = True
        self.postprocessing = False
        self.save_debug_pending = False

    def reset(self):
        self.paused = False
        self.decimate = 0
        self.postprocessing = False


state = AppState()

DEBUG_SAVE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data", "realsense_open3d"
)


def save_debug_bundle(
    depth_frame,
    color_image_rgb,
    colorized_depth_bgr,
    rs_points,
    mapped_frame,
    verts,
    texcoords,
    w,
    h,
):
    """保存深度 / 彩色 / RealSense PLY / 顶点网格。"""
    os.makedirs(DEBUG_SAVE_ROOT, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(DEBUG_SAVE_ROOT, f"snapshot_{ts}")

    depth_u16 = np.asanyarray(depth_frame.get_data())
    cv2.imwrite(f"{base}_depth_u16.png", depth_u16)
    np.save(f"{base}_depth_u16.npy", depth_u16)

    valid_d = depth_u16[depth_u16 > 0]
    if valid_d.size > 0:
        lo, hi = np.percentile(valid_d, [2, 98])
        vis = np.clip(depth_u16.astype(np.float32), lo, hi)
        vis = ((vis - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(depth_u16, dtype=np.uint8)
    vis_bgr = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(f"{base}_depth_colormap.png", vis_bgr)

    if color_image_rgb.ndim == 3 and color_image_rgb.shape[2] >= 3:
        color_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
    else:
        color_bgr = color_image_rgb
    cv2.imwrite(f"{base}_color_bgr.png", color_bgr)

    cv2.imwrite(f"{base}_realsense_depth_colorized.png", colorized_depth_bgr)

    rs_points.export_to_ply(f"{base}_realsense_tex.ply", mapped_frame)

    np.savez_compressed(
        f"{base}_verts_texcoords.npz",
        verts=verts,
        texcoords=texcoords,
        grid_w=w,
        grid_h=h,
    )

    n_valid = int(np.sum(depth_u16 > 0))
    meta = (
        f"pipeline: realsense_open3d (no rs.align; optional decimate + post filters)\n"
        f"decimate_step: {state.decimate} (magnitude 2^{state.decimate})\n"
        f"postprocessing: {state.postprocessing}\n"
        f"color_mode_rgb_not_depth_vis: {state.color}\n"
        f"depth_shape: {depth_u16.shape}\n"
        f"verts_grid_shape: {(h, w, 3)}\n"
        f"depth_valid_pixels: {n_valid}\n"
    )
    with open(f"{base}_meta.txt", "w", encoding="utf-8") as mf:
        mf.write(meta)

    print(f"[debug] 已写入 {base}_* → {DEBUG_SAVE_ROOT}")


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        return

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    other_stream, other_format = rs.stream.color, rs.format.rgb8
    config.enable_stream(other_stream, other_format, 30)

    pipeline.start(config)
    profile = pipeline.get_active_profile()

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()
    filters = [
        rs.disparity_transform(),
        rs.spatial_filter(),
        rs.temporal_filter(),
        rs.disparity_transform(False),
    ]

    vis_o3d = o3d.visualization.Visualizer()
    vis_o3d.create_window(window_name="RealSense PointCloud (Open3D)")
    pcd_o3d = o3d.geometry.PointCloud()
    first_o3d_frame = True

    cv2.namedWindow("color_image", cv2.WINDOW_NORMAL)

    print(__doc__)

    try:
        running = True
        while running:
            if state.paused:
                k = cv2.waitKey(50) & 0xFF
                if k == ord("q") or k == 27:
                    running = False
                elif k == ord("p"):
                    state.paused = False
                vis_o3d.poll_events()
                vis_o3d.update_renderer()
                continue

            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame().as_video_frame()
            other_frame = frames.first(other_stream).as_video_frame()

            depth_frame = decimate.process(depth_frame)

            if state.postprocessing:
                for f in filters:
                    depth_frame = f.process(depth_frame)

            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile
            ).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            color_image = np.asanyarray(other_frame.get_data())
            cv2.imshow("color_image", color_image)

            colorized_depth = colorizer.colorize(depth_frame)
            depth_colormap = np.asanyarray(colorized_depth.get_data())

            if state.color:
                mapped_frame, color_source = other_frame, color_image
            else:
                mapped_frame, color_source = colorized_depth, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            verts = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
            texcoords = np.asarray(points.get_texture_coordinates(2))

            verts_flat = verts.reshape(-1, 3)
            tex_flat = texcoords.reshape(-1, 2)
            ch, cw = color_source.shape[0], color_source.shape[1]
            u = np.clip((tex_flat[:, 0] * cw).astype(np.int32), 0, cw - 1)
            v_pix = np.clip((tex_flat[:, 1] * ch).astype(np.int32), 0, ch - 1)
            if state.color:
                px = color_source[v_pix, u]
                rgb = px.astype(np.float64) / 255.0
            else:
                bgr = color_source[v_pix, u]
                rgb = bgr[:, ::-1].astype(np.float64) / 255.0

            valid = np.isfinite(verts_flat).all(axis=1) & (verts_flat[:, 2] > 0)
            verts_show = verts_flat[valid]
            rgb_show = rgb[valid]

            pcd_o3d.points = o3d.utility.Vector3dVector(verts_show)
            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_show)
            if first_o3d_frame:
                vis_o3d.add_geometry(pcd_o3d)
                first_o3d_frame = False
            vis_o3d.update_geometry(pcd_o3d)
            vis_o3d.poll_events()
            vis_o3d.update_renderer()

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                running = False
            elif k == ord("p"):
                state.paused = True
            elif k == ord("r"):
                state.reset()
                decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
            elif k == ord("d"):
                state.decimate = (state.decimate + 1) % 3
                decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
            elif k == ord("c"):
                state.color ^= True
            elif k == ord("f"):
                state.postprocessing ^= True
            elif k == ord("s"):
                out_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite("out.png", out_bgr)
                print("已保存 out.png")
            elif k == ord("t"):
                state.save_debug_pending = True
            elif k == ord("e"):
                points.export_to_ply("./out.ply", mapped_frame)
                print("已导出 out.ply")

            if state.save_debug_pending:
                save_debug_bundle(
                    depth_frame,
                    color_image,
                    depth_colormap,
                    points,
                    mapped_frame,
                    verts,
                    texcoords,
                    w,
                    h,
                )
                state.save_debug_pending = False

    finally:
        try:
            vis_o3d.destroy_window()
        except Exception:
            pass
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == "__main__":
    main()
