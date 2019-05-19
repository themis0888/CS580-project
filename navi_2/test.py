import pyexr
import data

file = pyexr.open("samples/sample.exr")

print("Width:", file.width)
print("Height:", file.height)
print("Available channels:")
file.describe_channels()
    
print("Default channels:", file.channel_map['default'])
file = pyexr.open('samples/sample.exr')

for chan in ["default", "albedo", "diffuse", "specular", "normal", "normalVariance", "depth", "visibility"]:
    data.show_channel(file, chan)