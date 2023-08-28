## build

```
docker build -t qwen-7b-fast:latest --platform linux/amd64 . 
```

## run

```
docker run -it --gpus all -d --restart always -v /var/run/docker.sock:/var/run/docker.sock --name qwen-7b-fast -p 8080:8080 --user=20001:20001 --platform linux/amd64 qwen-7b-fast:latest
```

docker run -it --name qwen-7b-fast -p 8080:8080 --platform linux/amd64 qwen-7b-fast:latest