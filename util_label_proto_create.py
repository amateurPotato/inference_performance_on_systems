from object_detection.protos import string_int_label_map_pb2
import pandas as pd
from google.protobuf import text_format

df = pd.read_csv("C:\Users\aan22\Downloads\cars_data.csv")
labels = df["'label'"].unique().tolist()

pbm = string_int_label_map_pb2.StringIntLabelMap()

for idx, label in enumerate(labels[:-1]):
    pb = string_int_label_map_pb2.StringIntLabelMapItem()
    pb.id = idx+1
    pb.display_name = label
    pbm.item.append(pb)

f = open('labels.txt', 'w')
f.write(text_format.MessageToString(pbm))
f.close()
