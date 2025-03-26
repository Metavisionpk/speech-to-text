# *******Section 1 live speech to text using Amazon Transcribe********

import asyncio
import sounddevice
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent


"""
Here's an example of a custom event handler you can extend to
process the returned transcription results as needed. This
handler will simply print the text out to your interpreter.
"""
class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results
        for result in results:
        #   if not result.is_partial: # Only process final results
            for alt in result.alternatives:
                print(alt.transcript)


async def mic_stream():
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    # Be sure to use the correct parameters for the audio stream that matches
    # the audio formats described for the source language you'll be using:
    # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


async def write_chunks(stream):
    # This connects the raw audio chunks generator coming from the microphone
    # and passes them along to the transcription stream.
    async for chunk, status in mic_stream():
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()


async def basic_transcribe():
    # Setup up our client with our chosen AWS region
    client = TranscribeStreamingClient(region="us-east-1")

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm"
    )

    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(stream), handler.handle_events())
    
loop = asyncio.get_event_loop()
loop.run_until_complete(basic_transcribe())
loop.close()

#*******Section 2 speech to text by uploading audio file using Amazon Transcribe********

# import asyncio
# import librosa
# import numpy as np
# from amazon_transcribe.client import TranscribeStreamingClient
# from amazon_transcribe.handlers import TranscriptResultStreamHandler
# from amazon_transcribe.model import TranscriptEvent


# # Custom Event Handler for Processing Transcription Results
# class MyEventHandler(TranscriptResultStreamHandler):
#     async def handle_transcript_event(self, transcript_event: TranscriptEvent):
#         results = transcript_event.transcript.results
#         for result in results:
#             if not result.is_partial: # Only process final results
#                 for alt in result.alternatives:
#                     print("Transcribed Text:", alt.transcript)


# # Function to Read and Stream Audio File
# async def file_stream(audio_file):
#     # Load audio file
#     audio, sr = librosa.load(audio_file, sr=16000)  # Ensure 16kHz sample rate
#     audio_bytes = (np.array(audio) * 32767).astype(np.int16).tobytes()  # Convert to PCM 16-bit

#     # Stream audio in chunks
#     chunk_size = 1024 * 2  # Match AWS Transcribe's chunk size
#     for i in range(0, len(audio_bytes), chunk_size):
#         yield audio_bytes[i : i + chunk_size]  # Yield chunks of audio


# # Function to Send Audio to AWS Transcribe
# async def transcribe_audio(audio_file):
#     client = TranscribeStreamingClient(region="us-east-1")

#     # Start transcription stream
#     stream = await client.start_stream_transcription(
#         language_code="en-US",
#         media_sample_rate_hz=16000,
#         media_encoding="pcm",
#     )

#     # Start event handler
#     handler = MyEventHandler(stream.output_stream)

#     # Stream audio file and process transcription
#     await asyncio.gather(write_audio_chunks(stream, audio_file), handler.handle_events())


# # Function to Write Audio Chunks to AWS Transcribe
# async def write_audio_chunks(stream, audio_file):
#     async for chunk in file_stream(audio_file):
#         await stream.input_stream.send_audio_event(audio_chunk=chunk)
#     await stream.input_stream.end_stream()


# # Run Transcription with File Upload
# audio_file_path = "output.wav"  # Replace with your uploaded audio file
# loop = asyncio.get_event_loop()
# loop.run_until_complete(transcribe_audio(audio_file_path))
# loop.close()
