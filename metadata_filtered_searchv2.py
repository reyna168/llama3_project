from datasets import load_dataset
from PIL import Image

from pinecone import Pinecone
from pinecone import ServerlessSpec

import os


# initialize connection to pinecone
api_key = os.environ.get('PINECONE_API_KEY') or "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"

#pinecone_api_key="pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"

# configure client
#pc = Pinecone(api_key=api_key)



#下载数据集，查看数据
fashion = load_dataset(
    # 数据集名称
    "ashraq/fashion-product-images-small",
    # 数据集会分这不同的子集，如训练集train、验证集validation和测试集test,用于不同的阶段
    # split train 希望加载的是预训练部分
    split="train"
)

print(fashion)

# 取出所有的图片列
images = fashion['image']
# 移除image列后，就是文本列，我们交给了metadata
metadata = fashion.remove_columns('image')
# 显示其中一张图片，显示如下 
images[900]

# image = images[900]


# image.show()

#metada数据
# Hugging Face 的数据集调用to_pandas方法生成pandas的DataFrame
metadata = metadata.to_pandas()
# 显示前五条
metadata.head()

#从pinecone_text文本工具库的稀疏模块中引入BM25Encoder 编码工具
from pinecone_text.sparse import BM25Encoder
# 实例化bm25实例
bm25 = BM25Encoder()
# 训练一下，是为了让 BM25Encoder 对 `metadata` 数据集中 'productDisplayName' 列中的文本数据进行学习，计算文档中每个词的 IDF（逆文档频率）和其他相关统计量。
bm25.fit(metadata['productDisplayName'])
metadata['productDisplayName'][0]

# 对查询字符串进行编码
bm25.encode_queries(metadata['productDisplayName'][0])
# 对文档进行编码
bm25.encode_documents(metadata['productDisplayName'][0])

# 从HuggingFace的文本编码库SentenceTransformer库中引入sentence-transformers/clip-ViT-B-32  来自openai开源的clip
# 设备是否支持gpu计算
# 当下流行NLP框架
from torch import torch

from sentence_transformers import SentenceTransformer, util

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
model
# 也对productDisplayName 列进行clip编码
dense_vec = model.encode([metadata['productDisplayName'][0]])
# 返回向量维度
print(dense_vec.shape)

import torch
# 是否支持显卡
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# INDEX_NAME = 'dl-ai'
# initialize connection to pinecone
api_key = os.environ.get('PINECONE_API_KEY') or "pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"

#pinecone_api_key="pcsk_tc46z_6wbCqmtFErgvq29sqKPbLNpwcXDMMRkXnmVBYju1SvCcAezF3p7Gkta3Lhj2GHr"

# configure client
pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

# choose a name for your index
index_name = 'dl-ai'

import time

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=512,
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
# view index stats
index.describe_index_stats()

from tqdm.auto import tqdm
# 每批encode处理100个
batch_size = 100
# 总计1000个
fashion_data_num = 1000
# tqdm进度条显示 第一个参数使用range创建了一个可迭代对象
# min 取fashion数据集与1000 之间的较小者，
#batch_size 每次处理base_size个
for i in tqdm(range(0, min(fashion_data_num,len(fashion)), batch_size)):
    # find end of batch   i会自增的 怕最后一页不到batch_size
    i_end = min(i+batch_size, len(fashion))
    # extract metadata batch 把相应部分拿出来
    meta_batch = metadata.iloc[i:i_end]
    # 将dataframe转换为字典，orient="records"意思是按每一行记录转成字典
    meta_dict = meta_batch.to_dict(orient="records")
    # concatinate all metadata field except for id and year to form a single string
    # meta_batch.columns.isin(['id', 'year'])的意思是meta_batch列表里取id  year这两列
    #~按位取反操作符  那么就是除id year 两列外其余列都要
    # join 将这些列用空格连起来
    # 这样所有的文本列都在一起了， 做embedding 包含这行的所有语义
    meta_batch = [" ".join(x) for x in meta_batch.loc[:, ~meta_batch.columns.isin(['id', 'year'])].values.tolist()]
    # extract image batch  将这一批的图像列拿出来
    img_batch = images[i:i_end]
    # create sparse BM25 vectors   对文本内容做稀疏向量编码
    sparse_embeds = bm25.encode_documents([text for text in meta_batch])
    # create dense vectors
    # clip 密集模型用于图片的编码
    dense_embeds = model.encode(img_batch).tolist()
    # create unique IDs
    # 列表推导式 将从i  到 i_end 的数字i 拼到一起 代表这段数据也是唯一的
    ids = [str(x) for x in range(i, i_end)]

    upserts = []
    # loop through the data and create dictionaries for uploading documents to pinecone index
    # 将ids,sparse_embeds, .... 等元素用zip打包成一个元组。
    for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict):
        # 添加元素  
        upserts.append({
            'id': _id,
            # 这个之前的没有， sparse_values存的稀疏向量 
            'sparse_values': sparse,
            # values 值存的是图片编码
            'values': dense,
            # 媒体数据用的是meta_dict 本身的json 
            'metadata': meta
        })
    # upload the documents to the new hybrid index
    index.upsert(upserts)

# show index description after uploading the documents
index.describe_index_stats()

# 我们想查找深蓝色法国连体男士牛仔裤
query = "dark blue french connection jeans for men"
# 对查询文本做稀疏编码
sparse = bm25.encode_queries(query)
# 对查询文本再做密集编码
dense = model.encode(query).tolist()
# 返回相似的14条
# pinecone index支持 稀疏、密集向量同时查询
result = index.query(
    top_k=14,
    vector=dense,
    sparse_vector=sparse,
    include_metadata=True
)
#  在输出中拿出images
imgs = [images[int(r["id"])] for r in result["matches"]]

print(imgs)

from IPython.core.display import HTML
# 二进制流文件
from io import BytesIO
# 用于将图片转为base64
from base64 import b64encode

# function to display product images
def display_result(image_batch):
    figures = []
    for img in image_batch:
        # 二进制实例
        b = BytesIO()
        # 以png的格式存放
        img.save(b, format='png')
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="data:image/png;base64,{b64encode(b.getvalue()).decode('utf-8')}" style="width: 90px; height: 120px" >
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')


from PIL import Image, ImageDraw
#  # 創建一個空白畫布 
# canvas_width = 60 * 5 
# # 假設每行顯示5張圖片
# canvas_height = 80 * 3 
# # 假設顯示3行圖片
# canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
# draw = ImageDraw.Draw(canvas)


# # 將圖片依次放入畫布中
# for i, img in enumerate(imgs):
#      row = i // 5 
#      col = i % 5 
#      canvas.paste(img, (col * 60, row * 80)) # 顯示畫布
#      canvas.show()
# # 当我们在index.query()方法中除入传入vector外， 还传入了sparse_vector, 我们完成了一次混合向量查询
# # 

from PIL import Image

def create_image_canvas(image_list, rows, cols, image_size=(60, 80), spacing=10):
    """
    Combines multiple images into a single canvas.
    
    Parameters:
        image_list (list): List of PIL.Image objects.
        rows (int): Number of rows in the canvas.
        cols (int): Number of columns in the canvas.
        image_size (tuple): Size (width, height) of each image.
        spacing (int): Space between images.
    
    Returns:
        PIL.Image: Combined canvas with images.
    """
    # Calculate canvas size
    canvas_width = cols * (image_size[0] + spacing) - spacing
    canvas_height = rows * (image_size[1] + spacing) - spacing
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Place each image on the canvas
    for idx, img in enumerate(image_list):
        if idx >= rows * cols:
            break  # Stop if more images than grid space
        # Resize image
        img = img.resize(image_size)
        # Calculate position
        col = idx % cols
        row = idx // cols
        x = col * (image_size[0] + spacing)
        y = row * (image_size[1] + spacing)
        canvas.paste(img, (x, y))

    return canvas


# Example usage
from PIL import Image

# # Create some sample images
# image_list = [
#     Image.new("RGB", (60, 80), "red"),
#     Image.new("RGB", (60, 80), "blue"),
#     Image.new("RGB", (60, 80), "green"),
#     Image.new("RGB", (60, 80), "yellow"),
#     Image.new("RGB", (60, 80), "purple"),
#     Image.new("RGB", (60, 80), "orange")
# ]

# Create a canvas with 2 rows and 3 columns
canvas = create_image_canvas(imgs, rows=5, cols=3)

# Show the canvas
canvas.show()

# Save the canvas as a file (optional)
canvas.save("combined_canvas.png")
