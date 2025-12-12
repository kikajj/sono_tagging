import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors

st.set_page_config(page_title="B-mode + A-mode Ultrasound", layout="wide")
st.title("B-mode Ultrasound with Layer Tagging + Results")

uploaded = st.file_uploader("B-mode 초음파 이미지 업로드 (PNG/JPG)", type=["png","jpg","jpeg"])

# Thresholds
TH_BASELINE = 60
TH_BRIGHT = 150
TH_SUPERFICIAL = 150
TH_DEEP = 120

COLOR_BASELINE = "white"
COLOR_EPIDERMIS = "yellow"
COLOR_SUPERFICIAL = "magenta"
COLOR_DEEP = "cyan"
SCAN_LINE_COLOR = "red"

anchor_intensities = np.array([0,42,84,126,168,210,255],dtype=float)
anchor_colors = ["black","green","lightgreen","blue","red","yellow","white"]

def to_grayscale(np_img):
    if np_img.ndim==2: 
        return np_img.astype(np.float32)
    r,g,b = np_img[...,0], np_img[...,1], np_img[...,2]
    return (0.2126*r+0.7152*g+0.0722*b).astype(np.float32)

def build_smooth_colormap():
    positions = anchor_intensities/255.0
    return colors.LinearSegmentedColormap.from_list("anchored_grad", list(zip(positions,anchor_colors)))

def find_transitions_in_row(row_vals, run_len, W,
                            th_baseline=TH_BASELINE, th_bright=TH_BRIGHT,
                            th_superficial=TH_SUPERFICIAL, th_deep=TH_DEEP):
    start_search=100
    xb=None
    for i in range(start_search,W):
        if row_vals[i]>th_baseline:
            xb=i; break
    if xb is None: return None,None,None,None
    mm_per_pixel=10.0/W

    # Epidermis (lookahead 제외)
    xe=None; max_prev=row_vals[xb]
    for i in range(xb+run_len,W):
        max_prev=max(max_prev,row_vals[i-1])
        if row_vals[i]<th_bright or row_vals[i]<=max_prev/2:
            if (i-xb)*mm_per_pixel>=0.05:
                xe=i; break

    # Superficial dermis (강화된 lookahead)
    xs=None; sup_limit=int(1.0/mm_per_pixel)
    if xe is not None:
        for i in range(xe+run_len,min(xe+sup_limit,W)):
            prev_mean=np.mean(row_vals[max(0,i-10):i])
            if row_vals[i]>=th_superficial or (row_vals[i]-prev_mean)>=75:
                if (i-xe)*mm_per_pixel>=0.1:
                    lookahead=row_vals[i+1:i+11] if i+11<=W else row_vals[i+1:]
                    if np.any(np.diff(lookahead)<=-30): continue
                    extended=row_vals[i+1:i+16] if i+16<=W else row_vals[i+1:]
                    if np.any(np.diff(extended)>=100): continue
                    xs=i; break

    # Deep dermis (lookahead 유지)
    xd=None; deep_limit=int(2.0/mm_per_pixel)
    if xs is not None:
        for i in range(xs+run_len,min(xs+deep_limit,W)):
            if row_vals[i]<th_deep or abs(row_vals[i]-row_vals[i-1])>=75:
                if (i-xs)*mm_per_pixel>=0.3:
                    lookahead=row_vals[i+1:i+11] if i+11<=W else row_vals[i+1:]
                    if np.any(lookahead>=150): continue
                    xd=i; break
    return xb,xe,xs,xd

if uploaded is not None:
    img=Image.open(uploaded).convert("RGB")
    np_img=np.array(img)
    gray=to_grayscale(np_img)
    H,W=gray.shape
    y_pos=st.slider("A-mode 표시할 y좌표",0,H-1,H//2)

    row_vals=gray[y_pos,:]
    xb,xe,xs,xd=find_transitions_in_row(row_vals,5,W)

    baseline_points=[]; epidermis_points=[]; superficial_points=[]; deep_points=[]
    for y in range(H):
        xb_y,xe_y,xs_y,xd_y=find_transitions_in_row(gray[y,:],5,W)
        if xb_y is not None: baseline_points.append((xb_y,y))
        if xe_y is not None: epidermis_points.append((xe_y,y))
        if xs_y is not None: superficial_points.append((xs_y,y))
        if xd_y is not None: deep_points.append((xd_y,y))

    # 결과값 산출 (20pt 평균)
    mm_per_pixel = 10.0 / W
    ys = range(max(0,y_pos-10), min(H,y_pos+10))
    ratios = []
    dist_be, dist_bs, dist_bd = [], [], []
    for yy in ys:
        xb_y,xe_y,xs_y,xd_y=find_transitions_in_row(gray[yy,:],5,W)
        if xb_y is not None and xe_y is not None:
            dist_be.append((xe_y-xb_y)*mm_per_pixel)
        if xb_y is not None and xs_y is not None:
            dist_bs.append((xs_y-xb_y)*mm_per_pixel)
        if xb_y is not None and xd_y is not None:
            dist_bd.append((xd_y-xb_y)*mm_per_pixel)
        ratios.append(np.mean(gray[yy,:]<TH_BASELINE))
    low_ratio = np.mean(ratios)
    mean_be = np.mean(dist_be) if dist_be else None
    mean_bs = np.mean(dist_bs) if dist_bs else None
    mean_bd = np.mean(dist_bd) if dist_bd else None

    # 좌측: 사진, 우측: 결과값 + 박스
    col1, col2 = st.columns([2,1])

    with col1:
        fig_b,ax_b=plt.subplots(figsize=(W/40,H/120),dpi=300)
        ax_b.imshow(np_img, interpolation='nearest')
        if baseline_points:
            ax_b.plot([x for x,y in baseline_points],[y for x,y in baseline_points],
                      color=COLOR_BASELINE,linestyle="--",linewidth=1)
        if epidermis_points:
            ax_b.plot([x for x,y in epidermis_points],[y for x,y in epidermis_points],
                      color=COLOR_EPIDERMIS,linestyle="--",linewidth=1)
        if superficial_points:
            ax_b.plot([x for x,y in superficial_points],[y for x,y in superficial_points],
                      color=COLOR_SUPERFICIAL,linestyle="--",linewidth=1)
        if deep_points:
            ax_b.plot([x for x,y in deep_points],[y for x,y in deep_points],
                      color=COLOR_DEEP,linestyle="--",linewidth=1)
        ax_b.axhline(y=y_pos,color=SCAN_LINE_COLOR,linestyle="--",linewidth=1)
        ax_b.axis("off")
        st.pyplot(fig_b)

    with col2:
        st.markdown("### 결과값 (20pt 평균)")
        st.write(f"Low echogenic pixel ratio: {low_ratio:.3f}")
        if mean_be: st.write(f"Baseline ~ Epidermis: {mean_be:.3f} mm")
        if mean_bs: st.write(f"Baseline ~ Superficial dermis: {mean_bs:.3f} mm")
        if mean_bd: st.write(f"Baseline ~ Deep dermis: {mean_bd:.3f} mm")

        st.markdown("### 태깅 색상 설명")
        fig_leg, ax_leg = plt.subplots(figsize=(3,2))

        # Baseline
        ax_leg.plot([0,1],[4,4],color=COLOR_BASELINE,linewidth=3)
        ax_leg.text(1.2,4,"Baseline",va="center",fontsize=10)

        # Epidermis
        ax_leg.plot([0,1],[3,3],color=COLOR_EPIDERMIS,linewidth=3)
        ax_leg.text(1.2,3,"Epidermis",va="center",fontsize=10)

        # Superficial dermis
        ax_leg.plot([0,1],[2,2],color=COLOR_SUPERFICIAL,linewidth=3)
        ax_leg.text(1.2,2,"Superficial dermis",va="center",fontsize=10)

        # Deep dermis
        ax_leg.plot([0,1],[1,1],color=COLOR_DEEP,linewidth=3)
        ax_leg.text(1.2,1,"Deep dermis",va="center",fontsize=10)

        ax_leg.axis("off")
        st.pyplot(fig_leg)

else:
    st.info("B-mode 초음파 이미지를 업로드해 주세요.")