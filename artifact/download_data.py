import gdown

output = "SC24-result.zip"
id = "1G8A4HmpgxAJpGAqf6U6UgeaJmtGfjRSP"
gdown.download(id=id, output=output)

# unzip
import zipfile
with zipfile.ZipFile(output, 'r') as zip_ref:
    # directly extract all files
    zip_ref.extractall()