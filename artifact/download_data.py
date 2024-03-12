import gdown

output = "SC24-results.zip"
id = "1risb7I4Cg6NPXHYYGgsKHynU1kebsf5c"
gdown.download(id=id, output=output)

# unzip
import zipfile
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("SC24-results")