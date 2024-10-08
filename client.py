from gradio_client import Client, handle_file
import torch
import argparse
import os
import wget

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="http://101.126.90.71:50183/", type=str)
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--pose", required=True, type=str)
    parser.add_argument("--output", default="output", type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    client = Client(args.address)

    result = client.predict(api_name="/restart")

    step1_result = client.predict(
        video_input={"video": handle_file(args.video)},
        api_name="/get_frames_from_video",
    )
    print("Step 1 Result".center(100, "="))
    print(step1_result)

    step2_1_result = client.predict(image_selection_slider=1, api_name="/select_image")
    print("Step 2.1 Result".center(100, "="))
    print(step2_1_result)

    step2_2_result = client.predict(
        track_pause_number_slider=int(step1_result[4]["value"] - 1),
        api_name="/end_image",
    )

    print("Step 2.2 Result".center(100, "="))
    print(step2_2_result)

    model = torch.load(args.pose)
    coors = model[0][[0, 11, 12, 13, 14]][:, :2] * step1_result[1]

    step2_3_result = client.predict(
        point_prompt="Positive",
        coors={
            "headers": ["Coor X", "Coor Y"],
            "data": coors.tolist(),
            "metadata": None,
        },
        api_name="/sam_refine_click",
    )

    print("Step 2.3 Result".center(100, "="))
    print(step2_3_result)

    step2_4_result = client.predict(mask_dropdown=[], api_name="/add_multi_mask")

    print("Step 2.4 Result".center(100, "="))
    print(step2_4_result)

    step3_1_result = client.predict(
        mask_dropdown=step2_4_result[0]["value"], api_name="/vos_tracking_video"
    )

    print("Step 3.1 Result".center(100, "="))
    print(step3_1_result)

    step3_2_result = client.predict(
        resize_ratio_number=1,
        dilate_radius_number=8,
        raft_iter_number=20,
        subvideo_length_number=80,
        neighbor_length_number=10,
        ref_stride_number=10,
        mask_dropdown=step2_4_result[0]["value"],
        api_name="/inpaint_video",
    )

    print("Step 3.2 Result".center(100, "="))
    print(step3_2_result)

    video_path = args.address + "file=" + step3_2_result[0]["video"]
    wget.download(video_path, out=args.output)
