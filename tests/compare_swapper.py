# 对比 Rust 和 Python 的 swapper 中间输出
import cv2, numpy as np

# 重新运行 Python 基准，保存 swapper 输出
M = r"C:\Users\Administrator\MagicMirror\models"
SIZE = 640; SCORE_THRESH = 0.1; NMS_THRESH = 0.5
import onnxruntime as ort, onnx

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

scrfd = ort.InferenceSession(f"{M}\\scrfd_2.5g.onnx")
arcface = ort.InferenceSession(f"{M}\\arcface_w600k_r50.onnx")
swapper = ort.InferenceSession(f"{M}\\inswapper_128_fp16.onnx")
onnx_model = onnx.load(f"{M}\\inswapper_128_fp16.onnx")
weight = onnx.numpy_helper.to_array(onnx_model.graph.initializer[-1])
INSWAPPER_TMPL = np.array([[0.36167656,0.40387734],[0.63696719,0.40235469],[0.50019687,0.56044219],[0.38710391,0.72160547],[0.61507734,0.72034453]])
ARCFACE_TMPL = np.array([[0.34191607,0.46157471],[0.65653393,0.45983393],[0.500225,0.64050536],[0.370975,0.57523],[0.63152143,0.57341857]])

def preprocess(img):
    h,w=img.shape[:2]; scale=min(SIZE/w,SIZE/h); nw,nh=int(round(w*scale)),int(round(h*scale))
    r=cv2.resize(img,(nw,nh)); c=np.full((SIZE,SIZE,3),127.5,np.uint8)
    px,py=(SIZE-nw)//2,(SIZE-nh)//2; c[py:py+nh,px:px+nw]=r
    b=c[:,:,::-1].astype(np.float32); b=(b-127.5)/128.0
    return b.transpose(2,0,1)[np.newaxis,...].astype(np.float32),scale,px,py

def gen_anchors(stride,at):
    fh=SIZE//stride; fw=SIZE//stride; a=[]
    for i in range(fh):
        cy=(fh-1-i)*stride
        for j in range(fw):
            cx=j*stride
            for _ in range(at): a.append((cx,cy))
    return np.array(a,np.float32)

def detect(img):
    b,sc,px,py=preprocess(img); outs=scrfd.run(None,{"input":b}); all_b=[]
    for si,st in enumerate([8,16,32]):
        scs=outs[si].reshape(-1); bx=outs[si+3].reshape(-1,4); kp=outs[si+6].reshape(-1,10)
        a=gen_anchors(st,2)
        for i in range(len(scs)):
            if scs[i]<SCORE_THRESH: continue
            cx,cy=a[i]; l,t,r,b2=bx[i]*st
            x1=(cx-l-px)/sc; y1=(cy-t-py)/sc; x2=(cx+r-px)/sc; y2=(cy+b2-py)/sc
            ldm=np.zeros((5,2))
            for j in range(5):
                ldm[j,0]=(cx+kp[i,j*2]*st-px)/sc; ldm[j,1]=(cy+kp[i,j*2+1]*st-py)/sc
            all_b.append((scs[i],x1,y1,x2,y2,ldm))
    all_b.sort(key=lambda x:-x[0]); keep=[]
    for b in all_b:
        if all(iou(b,k)<=NMS_THRESH for k in keep): keep.append(b)
    return keep

def iou(a,b):
    x1=max(a[1],b[1]);y1=max(a[2],b[2]);x2=min(a[3],b[3]);y2=min(a[4],b[4])
    inter=max(0,x2-x1)*max(0,y2-y1);ua=(a[3]-a[1])*(a[4]-a[2])+(b[3]-b[1])*(b[4]-b[2])-inter
    return inter/ua if ua>0 else 0

def warp_face(img,ldm,tmpl,size):
    t=tmpl*size; M=cv2.estimateAffinePartial2D(ldm.astype(np.float32),t.astype(np.float32),method=cv2.RANSAC,ransacReprojThreshold=100)[0]
    return cv2.warpAffine(img,M,(size,size),borderMode=cv2.BORDER_REPLICATE,flags=cv2.INTER_AREA),M

a=cv2.imread(os.path.join(FIXTURES, "a.jpg")); b=cv2.imread(os.path.join(FIXTURES, "b.png"))
af=detect(a)[0]; bf=detect(b)[0]

# Python warp 和 swapper
warped_a, M = warp_face(a, af[5], INSWAPPER_TMPL, 128)
warped_b, _ = warp_face(b, bf[5], ARCFACE_TMPL, 112)
blob = warped_b[:,:,::-1].astype(np.float32); blob = (blob-127.5)/128.0
blob = blob.transpose(2,0,1)[np.newaxis,...].astype(np.float32)
emb = arcface.run(None,{"input":blob})[0][0]; emb = emb / np.linalg.norm(emb)
temb = emb @ weight / np.linalg.norm(emb)
input_target = warped_a[:,:,::-1].transpose(2,0,1)[np.newaxis,...].astype(np.float32)/255.0
input_source = temb.reshape(1,-1).astype(np.float32)
result = swapper.run(None,{"target":input_target,"source":input_source})[0][0]
swapped_face = result.transpose(1,2,0).clip(0,1)[:,:,::-1]*255
swapped_face = swapped_face.astype(np.uint8)
cv2.imwrite(r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "py_swapped.jpg")", swapped_face)

# 读取 Rust 的 swapper 输出（需要从 Rust 获取）
# 保存 M 矩阵供对比
np.save(os.path.join(FIXTURES, "py_affine.npy"), M)
print(f"Python affine M:\n{M}")
print(f"Python swapped face saved: {swapped_face.shape}")
print(f"Python file size: {cv2.imencode('.jpg',swapped_face)[1].size} bytes")