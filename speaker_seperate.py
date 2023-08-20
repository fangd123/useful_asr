from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")

# 4. apply pretrained pipeline
diarization = pipeline("1.wav")

# 5. print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start}s stop={turn.end}s speaker_{speaker}")