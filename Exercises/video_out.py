from moviepy.editor import VideoFileClip
def process_movie(input_path, output_path,duration):
    clip = VideoFileClip(input_path)
    if duration > 0:
        out_clip = clip.fl_image(trkr.process).subclip(0,duration)
    else:
        out_clip = clip.fl_image(trkr.process)
    out_clip.write_videofile(output_path, audio=False)
process_movie("../project_video.mp4", "../project_video_out.mp4",-1)