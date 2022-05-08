# MSC-Final-Project-ST
## Docker container
To run it through a container follow these instructions:
- Open terminal on the directory path
- Build and start the container: `docker-compose up -d`
- Open your internet browser (Chrome / Safari / ...)
- Enter the Jupyter notebook URL: `http://localhost:8888`
- Open the relevant notebook from the "notebooks" folder or create a new one <b>(use the "FPST kernel" kernel)</b>. 

## STlearn Web APP
To use the STlearn web app there is a need in additional steps after you start the docker
1. Find the container ID: `docker ps` -> copy the value under "CONTAINER ID"
2. SShing the container: `docker exec -it {CONTAINER_ID} bash`
3. Active the web app within the container (container terminal): `stlearn launch`
4. Enter the web app URL: `http://127.0.0.1:5000/`
