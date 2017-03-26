vehicle_trkr=VehicleTracker2(test_img.shape,heatmap_threshold=15)
from moviepy.editor import VideoFileClip
def process_movie(input_path, output_path,duration):
    clip = VideoFileClip(input_path)
    if duration > 0:
        out_clip = clip.fl_image(vehicle_trkr.process).subclip(5,20)
    else:
        out_clip = clip.fl_image(vehicle_trkr.process)
    out_clip.write_videofile(output_path, audio=False)

process_movie("../project_video.mp4", "../project_video_out.mp4",5)