# CVPR 2022 - 3D Matching

You may want to configure some parameters in `src/config.py` before buid.

## Run in Docker

```bash
# Clone the repository.
git clone https://github.com/arkhodakov/cvpr-2022-matching
cd cvpr-2022-matching

# Build Docker image.
docker build -t cvpr-2022-matching .

# Run image and evaluate.
# NOTE: Put all your JSON data in ./data/.
docker run \
    -v {full/path/to/data/directory/}:/data/ \
    -v {full/path/to/output/directory/}:/code/output \
    -it cvpr-2022-matching /bin/bash

python main.py ../data/{reference_model}.json ../data/{user_model}.json --output/match.json

# Example:
python main.py ../data/01_OfficeLab01_Allfloors_columns.json ../data/01_OfficeLab01_Allfloors_users_columns.json --output output/match.json
```
