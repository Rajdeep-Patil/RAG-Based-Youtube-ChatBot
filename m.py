from youtube_transcript_api import YouTubeTranscriptApi
yt_api = YouTubeTranscriptApi()
video_id = 'J-KKHoEQRHA'
try:
    video_transcript = yt_api.fetch(video_id=video_id,languages=['en'])

    transcript = " ".join(text.text for text in video_transcript)
    print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")