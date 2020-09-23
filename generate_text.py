from functions import generate_text
import datetime
from LyricsModel import TrainedLyricsModel

date = datetime.datetime.now()

filepath = "weights.hdf5"

model = TrainedLyricsModel(filepath)


songFileName = "results/output_"+str(date)+".txt"
with open(songFileName, 'w') as f:
    f.write(generate_text(model, 500, 0.2))

print(generate_text(500, 0.2))