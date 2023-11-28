# 镜像image
## 1. 列出本地主机上的镜像,
REPOSITORY：表示镜像的仓库源,TAG：镜像的标签,IMAGE ID：镜像ID,CREATED：镜像创建时间,SIZE：镜像大小,同一仓库源可以有多个 TAG，代表这个仓库源的不同个版本

` docker images `

## 2.指定版本的镜像来启动运行容器,可以选择直接进入容器并启动命令
i: 交互式操作, -t: 终端, ubuntu:15.10: 这是指用 ubuntu 15.10 版本镜像为基础来启动容器, /bin/bash：放在镜像名后的是命令，这里我们希望有个交互式 Shell，因此用的是 /bin/bash , -d后台启动,-p端口映射，第一个是外部端口，第二个是docker内部端口，把内部的端口映射到外部端口,--name容器名称
`docker run -t -i --name test ubuntu:15.10 /bin/bash`
`docker run ubuntu:15.10 /bin/echo "Hello world"`
`docker run -itd ubuntu:15.10 /bin/sh -c "while true; do echo hello world; sleep 1; done"`
`docker run -d -p 5000:5000 training/webapp python app.py`
`docker run -itd -p 8505:8505 --name test_yuheng yuheng:0.2.0`
`docker port bf08b7f2cd89`

## 3.获取/下载/查找/删除镜像
`docker pull ubuntu:13.10`
`docker search httpd`
`docker rmi hello-world`


## 4.使用容器来更新镜像
-m: 提交的描述信息；-a: 指定镜像作者；e218edb10161：容器 ID；runoob/ubuntu:v2: 指定要创建的目标镜像名
`docker commit -m="has update" -a="runoob" e218edb10161 runoob/ubuntu:v2`

## 5. 使用Dockerfile从零构建一个新的镜像
Dockerfile 是一个用来构建镜像的文本文件，文本内容包含了一条条构建镜像所需的指令和说明,每一个指令都会在镜像上创建一个新的层，每一个指令的前缀都必须是大写的； 第一条FROM，指定使用哪个镜像源 ；RUN 指令告诉docker 在镜像内执行命令，安装了什么。。。 ；然后，我们使用 Dockerfile 文件，通过 docker build 命令来构建一个镜像。
-t ：指定要创建的目标镜像名; . ：Dockerfile 文件所在目录，可以指定Dockerfile 的绝对路径
`docker build -t runoob/centos:6.7 .`

`docker build -t yuheng:0.1.3 -f ./Dockerfile --no-cache .`
Dockerfile 的指令每执行一次都会在 docker 上新建一层。所以过多无意义的层，会造成镜像膨胀过大。
Dockerfile
```
FROM centos
FROM centos
RUN yum -y install wget \
    && wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
    && tar -xvf redis.tar.gz

```
```
FROM python:3.9
RUN pip install --upgrade pip -i https://repo.huaweicloud.com/repository/pypi/simple/
RUN mkdir -p /yuheng_project
# Install packages
ADD . /yuheng_project/
WORKDIR /yuheng_project
#使用cache 
RUN --mount=type=cache,mode=0777,target=/root/.cache/  pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ \
    && pip install -r ./requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN cp /yuheng_project/data/font/SimHei.ttf /usr/local/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf

RUN chmod -R 777 /yuheng_project/start.sh
WORKDIR ./yuheng_project
CMD ["/yuheng_project/start.sh"]
```

FROM 基于的镜像；RUN：用于执行后面跟着的命令行命令（RUN <命令行命令> / RUN ["可执行文件", "参数1", "参数2"]）；&& 符号连接命令，这样执行后，只会创建 1 层镜像；上下文路径，是指 docker 在构建镜像，有时候想要使用到本机的文件（比如复制），docker build 命令得知这个路径后，会将路径下的所有内容打包；COPY 复制指令，从外部的上下文目录中复制文件或者目录到容器里指定路径；COPY [--chown=<user>:<group>] <源路径1>...  <目标路径>。~~ADD~~
CMD 为启动的容器指定默认要运行的程序，程序运行结束，容器也就结束，类似于 RUN 指令，用于运行程序，但二者运行的时间点不同:CMD 在docker run 时运行。 RUN 是在 docker build。如果 Dockerfile 中如果存在多个 CMD 指令，仅最后一个生效
CMD <shell 命令> 
CMD ["<可执行文件或命令>","<param1>","<param2>",...] 
CMD ["<param1>","<param2>",...]  # 该写法是为 ENTRYPOINT 指令指定的程序提供默认参数
推荐使用第二种格式，执行过程比较明确。第一种格式实际上在运行的过程中也会自动转换成第二种格式运行，并且默认可执行文件是 sh。

ENTRYPOINT
类似于 CMD 指令，但其不会被 docker run 的命令行参数指定的指令所覆盖，而且这些命令行参数会被当作参数送给 ENTRYPOINT 指令指定的程序。

但是, 如果运行 docker run 时使用了 --entrypoint 选项，将覆盖 ENTRYPOINT 指令指定的程序。

优点：在执行 docker run 的时候可以指定 ENTRYPOINT 运行所需的参数。

注意：如果 Dockerfile 中如果存在多个 ENTRYPOINT 指令，仅最后一个生效。

可以搭配 CMD 命令使用：一般是变参才会使用 CMD ，这里的 CMD 等于是在给 ENTRYPOINT 传参，以下示例会提到。

示例：

假设已通过 Dockerfile 构建了 nginx:test 镜像：

FROM nginx

ENTRYPOINT ["nginx", "-c"] # 定参
CMD ["/etc/nginx/nginx.conf"] # 变参 



## 6. 设置镜像标签
tag 相当于给一个镜像打标签，这个镜像可以有多个tag，相当于多条绳子吊起一个球，所以不把所有tag删除，这个镜像就不会删除，所以可以随意使用tag进行版本docker管理
`docker tag 860c279d2fec runoob/centos:dev`
`docker tag yuheng:0.1.3 swr.cn-southwest-2.myhuaweicloud.com/pharmcube/cplatform/yuheng:0.1.3
`


# 容器container
## 1. 查看容器/日志/停止/启动/重新启动/删除
`docker ps`
`docker logs 2b1b7a428627` 
`docker stop test_yuheng`
`docker start b750bbbcfd88`
`docker restart <容器 ID>`
`docker rm -f 1e560fca3906`

## 2.进入后台的容器
~~docker attach 1e560fca3906 ~~如果从这个容器退出，会导致容器的停止~~~~

`docker exec -it 243c32535da7 /bin/bash` 如果从这个容器退出，容器不会停止

## 3. 导入和导出容器
`docker export 1e560fca3906 > ubuntu.tar`
`cat docker/ubuntu.tar | docker import - test/ubuntu:v1`
` docker import http://example.com/exampleimage.tgz example/imagerepo`


`docker push swr.cn-southwest-2.myhuaweicloud.com/pharmcube/cplatform/yuheng:0.1.3`

