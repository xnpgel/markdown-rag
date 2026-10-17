如何编译运行源码
本项目由python编写，无需编译。
运行方法：
1.安装conda环境
2.进入conda环境，使用pip install -r requirements.txt安装python相关依赖包。
3.在系统中安装好Milvus数据库和Ollama模型，配置config.py中的Milvus数据库端口和Ollama模型的端口。申请百炼API-key。
4.先在终端输入export DASHSCOPE_API_KEY="key"，其中key是百炼API-key。之后，进入conda环境，运行python ingest.py，程序切片md文件并存入Milvus向量数据库。之后运行python show_data.py，程序检查是否数据正确存入数据库。之后运行python query.py "question"，其中question是想要问的问题，就可以看到经过RAG提取的数据和大模型的回答了，回答标注了来源，方便检查。
5.最后，运行python delete_collection.py可以清理当前Milvus的数据库，避免之后更换数据后产生干扰。
注：可以运行python noRAG.py查看模型在没有RAG下的回答
