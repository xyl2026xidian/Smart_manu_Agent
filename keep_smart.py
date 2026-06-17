# ============================================================
# 文件名: sme_agent.py
# 说明: 智能制造工程导论 课程智能体 (Streamlit 部署)
# 功能: 课程概述 + 术语查询(30+术语) + 数字孪生实验室(4场景) + 知识问答(20+问题)
# 运行: streamlit run sme_agent.py
# ============================================================
# 作者: 智能制造课程组
# 版本: 2.0
# 更新日期: 2026年
# ============================================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import lti, step
import time
import random

# ======================= 页面配置 =======================
st.set_page_config(
    page_title="智能制造工程导论",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= 自定义CSS样式 =======================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1a3a5c; }
    .sub-header { font-size: 1.5rem; font-weight: 600; color: #2c5f8a; margin-top: 20px; }
    .metric-card { background: #f0f4f8; padding: 15px; border-radius: 10px; text-align: center; }
    .info-box { background: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2c5f8a; }
    .success-box { background: #e6f7e6; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; }
    .warning-box { background: #fff8e6; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; }
    .stButton>button { width: 100%; background-color: #2c5f8a; color: white; }
    .stButton>button:hover { background-color: #1a3a5c; }
</style>
""", unsafe_allow_html=True)

# ======================= 扩展术语库 (30+术语) =======================
TERMS = {
    # ————— 人工智能 —————
    "MLP": {
        "全称": "多层感知机 (Multilayer Perceptron)",
        "类别": "人工智能",
        "定义": "一种前馈神经网络，由输入层、多个隐藏层和输出层组成，通过反向传播训练，是深度学习的基础模型。",
        "应用": "用于工艺参数优化、设备故障预测、质量分类等。",
        "相关技术": "反向传播、激活函数、梯度下降"
    },
    "CNN": {
        "全称": "卷积神经网络 (Convolutional Neural Network)",
        "类别": "人工智能",
        "定义": "一种专门处理网格状数据（如图像）的深度学习模型，通过卷积层、池化层自动提取特征。",
        "应用": "产品表面缺陷检测、工件识别与分类、视觉引导机器人。",
        "相关技术": "卷积核、池化、特征图"
    },
    "RNN": {
        "全称": "循环神经网络 (Recurrent Neural Network)",
        "类别": "人工智能",
        "定义": "具有循环连接的神经网络，能够处理序列数据，记忆历史信息。",
        "应用": "设备剩余寿命预测、工艺时序数据建模。",
        "相关技术": "隐藏状态、梯度消失、序列建模"
    },
    "LSTM": {
        "全称": "长短期记忆网络 (Long Short-Term Memory)",
        "类别": "人工智能",
        "定义": "一种特殊的RNN，通过门控机制解决长期依赖问题，能记住长时间跨度的信息。",
        "应用": "设备振动信号分析、刀具磨损预测、能源消耗预测。",
        "相关技术": "遗忘门、输入门、输出门、细胞状态"
    },
    "GAN": {
        "全称": "生成对抗网络 (Generative Adversarial Network)",
        "类别": "人工智能",
        "定义": "由生成器和判别器组成的对抗式训练框架，可生成逼真的合成数据。",
        "应用": "缺陷样本生成（解决数据不平衡问题）、产品设计合成。",
        "相关技术": "对抗训练、生成器、判别器"
    },
    "Transformer": {
        "全称": "Transformer 模型",
        "类别": "人工智能",
        "定义": "基于自注意力机制的深度学习架构，可并行处理序列数据，是大模型的基础。",
        "应用": "工业知识问答、智能排产、生成式设计。",
        "相关技术": "自注意力、多头注意力、位置编码"
    },
    "工业大模型": {
        "全称": "工业大模型 (Industrial Foundation Model)",
        "类别": "人工智能",
        "定义": "面向工业领域的大规模预训练模型，融合多源工业数据，具备理解、推理、生成能力。",
        "应用": "智能排产、工艺知识问答、设备故障诊断、代码生成。",
        "相关技术": "预训练、微调、工业知识图谱"
    },
    "工业智能体": {
        "全称": "工业智能体 (Industrial Agent)",
        "类别": "人工智能",
        "定义": "基于大模型的自主智能系统，能感知环境、规划任务、调用工具、执行动作。",
        "应用": "智能调度、自动化质检、异常处理、协同制造。",
        "相关技术": "感知-决策-执行闭环、多智能体协同"
    },
    "机器视觉": {
        "全称": "机器视觉 (Machine Vision)",
        "类别": "人工智能",
        "定义": "通过工业相机和图像处理算法，模拟人类视觉功能，实现自动检测和识别。",
        "应用": "缺陷检测、尺寸测量、条码识别、机器人引导。",
        "相关技术": "图像处理、特征提取、深度学习"
    },
    "数字孪生": {
        "全称": "数字孪生 (Digital Twin)",
        "类别": "前沿技术",
        "定义": "物理实体在虚拟空间中的高保真映射模型，实现物理-虚拟的双向数据交互与实时同步。",
        "应用": "产线虚拟调试、设备预测性维护、加工过程仿真。",
        "相关技术": "数据采集、模型降阶、实时映射"
    },
    "具身智能": {
        "全称": "具身智能 (Embodied AI)",
        "类别": "前沿技术",
        "定义": "智能体通过物理实体与环境实时交互，实现感知、认知、决策和行动一体化。",
        "应用": "人形机器人、具身智能工业机器人、通用操作智能体。",
        "相关技术": "感知-行动闭环、物理交互、强化学习"
    },
    "人形机器人": {
        "全称": "人形机器人 (Humanoid Robot)",
        "类别": "前沿技术",
        "定义": "模仿人体形态和功能的机器人，具备双足行走、双臂操作等能力，可适应人类环境。",
        "应用": "工业场景实训、危险环境作业、柔性装配、物流搬运。",
        "相关技术": "双足平衡、灵巧手、运动规划"
    },
    # ————— 工业控制 —————
    "PLC": {
        "全称": "可编程逻辑控制器 (Programmable Logic Controller)",
        "类别": "工业控制",
        "定义": "工业现场的核心控制器，通过循环扫描执行逻辑、时序、计数、运算等指令，控制机械设备。",
        "应用": "生产线自动化控制、机器人协同、数据采集。",
        "相关技术": "梯形图、扫描周期、I/O模块"
    },
    "SCADA": {
        "全称": "监控与数据采集系统 (Supervisory Control and Data Acquisition)",
        "类别": "工业控制",
        "定义": "用于远程监控和采集分布式设备数据的系统，提供人机界面、报警管理、历史趋势分析。",
        "应用": "工厂设备状态监控、异常报警、能源管理。",
        "相关技术": "RTU、HMI、OPC UA"
    },
    "DCS": {
        "全称": "分布式控制系统 (Distributed Control System)",
        "类别": "工业控制",
        "定义": "面向流程工业（化工、炼油、电力）的大型控制系统，采用'分散控制、集中管理'的设计理念。",
        "应用": "炼油厂全流程控制、电厂机组协调控制、化工厂反应釜管理。",
        "相关技术": "冗余设计、PID控制、流程优化"
    },
    "CNC": {
        "全称": "计算机数控系统 (Computer Numerical Control)",
        "类别": "工业控制",
        "定义": "通过计算机控制机床运动，实现高精度、高速度的自动化加工。",
        "应用": "铣削加工、车削加工、五轴联动加工。",
        "相关技术": "插补算法、G代码、伺服控制"
    },
    # ————— 工业软件 —————
    "MES": {
        "全称": "制造执行系统 (Manufacturing Execution System)",
        "类别": "工业软件",
        "定义": "连接企业计划层（ERP）与控制层（PLC/SCADA）的桥梁，管理车间生产调度、质量追溯、设备监控。",
        "应用": "生产工单排程、质量过程管控、设备效率分析。",
        "相关技术": "实时数据库、工单管理、SPC"
    },
    "ERP": {
        "全称": "企业资源计划 (Enterprise Resource Planning)",
        "类别": "工业软件",
        "定义": "覆盖企业全业务链的管理软件，整合财务、采购、库存、销售、人力资源等资源。",
        "应用": "物料需求计划、成本核算、供应链管理。",
        "相关技术": "MRP、财务一体化、供应链协同"
    },
    "PLM": {
        "全称": "产品生命周期管理 (Product Lifecycle Management)",
        "类别": "工业软件",
        "定义": "管理产品从概念设计到退市全生命周期的数据与流程，是产品数据的源头。",
        "应用": "BOM管理、CAD数据管理、工程变更管理。",
        "相关技术": "配置管理、工作流引擎、协同设计"
    },
    "BOM": {
        "全称": "物料清单 (Bill of Materials)",
        "类别": "工业软件",
        "定义": "描述产品结构的数字化清单，包含零部件、用量、层级关系，是PLM/MES/ERP协同的数据核心。",
        "应用": "从设计BOM(EBOM)到制造BOM(MBOM)的转换驱动生产采购。",
        "相关技术": "EBOM、MBOM、BOM版本管理"
    },
    "工业互联网": {
        "全称": "工业互联网 (Industrial Internet)",
        "类别": "网络技术",
        "定义": "将工业设备、系统、人员通过物联网和互联网连接，实现数据采集、分析、优化的基础设施。",
        "应用": "设备联网、数据中台、跨厂区协同制造。",
        "相关技术": "IoT、5G、TSN、边缘计算"
    },
    "边缘计算": {
        "全称": "边缘计算 (Edge Computing)",
        "类别": "网络技术",
        "定义": "在靠近数据源（设备、传感器）一侧进行计算，减少数据传输延迟，保障实时性和安全性。",
        "应用": "车间级AI推理、设备实时控制、预测性维护。",
        "相关技术": "云边协同、轻量化AI、5G URLLC"
    },
    "数字主线": {
        "全称": "数字主线 (Digital Thread)",
        "类别": "数字化技术",
        "定义": "贯穿产品全生命周期的端到端数据流，将设计、制造、运维各阶段的数据无缝连接。",
        "应用": "西飞运-20数字化装配、产品全生命周期追溯。",
        "相关技术": "数据集成、端到端追溯、MBSE"
    },
    "HCPS": {
        "全称": "人-信息-物理系统 (Human-Cyber-Physical Systems)",
        "类别": "基础理论",
        "定义": "智能制造的核心理论框架，强调人、信息系统、物理系统的深度融合与协同。",
        "应用": "智能工厂设计、人机协作、制造系统架构。",
        "相关技术": "系统集成、人机交互、CPS"
    },
}

# ======================= 侧边栏导航 =======================
st.sidebar.image("https://via.placeholder.com/300x80?text=智能制造", use_column_width=True)
st.sidebar.title("📚 课程导航")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "选择模块",
    ["🏠 课程概述", "🔍 术语查询", "💻 数字孪生实验室", "❓ 知识问答"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("🏭 智能制造工程导论")
st.sidebar.caption("📅 2026年版 | v2.0")
st.sidebar.caption(f"📖 术语库: {len(TERMS)} 个")

# ======================= 1. 课程概述 =======================
if page == "🏠 课程概述":
    st.markdown('<div class="main-header">🏠 智能制造工程导论</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px">
    <h3>📌 课程定位</h3>
    <p>本课程面向本科三年级机械类专业学生，系统讲授智能制造的基本概念、关键技术和典型应用。
    通过理论学习与数字孪生实践相结合，培养学生对智能制造系统的整体认知与工程分析能力。</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📚 核心内容
        - **模块一：绪论** — 制造强国战略、HCPS 人-信息-物理系统
        - **模块二：关键技术** — 数字化、网络化、智能化
        - **模块三：智能装备** — 工业机器人、数控机床、AGV
        - **模块四：智能工厂** — PLM/MES/ERP、数字孪生
        - **模块五：智能产品** — 个性化定制、服务型制造
        - **模块六：案例研讨** — 汽车/工程机械/航空航天
        - **模块七：前沿展望** — 具身智能、大模型、工业智能体
        """)
    with col2:
        st.markdown("""
        ### 🎯 学习目标
        - 理解智能制造的核心概念与内涵
        - 掌握关键赋能技术的原理与应用
        - 了解智能装备与智能工厂的系统集成
        - 具备初步的智能制造系统分析能力
        - 能够通过数字孪生实践理解物理-虚拟映射
        """)
    
    st.markdown("---")
    st.markdown("### 📊 产业数据速览")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        st.metric("工业机器人产量 (2025)", "77.3万套", "↑28.0%")
    with col4:
        st.metric("智能工厂(领航级)", "15家", "首批")
    with col5:
        st.metric("工业互联网规模", "1.6万亿", "↑12%")
    with col6:
        st.metric("术语总数", f"{len(TERMS)}个", "持续更新")
    
    st.markdown("---")
    st.info("💡 **数字孪生实验室**：包含 3D机械臂交互、弹性末端动力学、梁变形分析、切削加工仿真四大模块，体验“物理→虚拟→交互”的完整孪生闭环。")
    
    with st.expander("📖 课程思政融入", expanded=False):
        st.markdown("""
        - **制造强国战略**：通过中国智能制造产业成就，增强民族自豪感
        - **自主创新精神**：通过国产工业机器人、数控机床案例，培养创新意识
        - **工匠精神**：通过精密制造和数字孪生实践，培养精益求精的作风
        - **科技报国**：通过前沿技术展望，激发学生投身制造强国建设
        """)

# ======================= 2. 术语查询 =======================
elif page == "🔍 术语查询":
    st.markdown('<div class="main-header">📖 智能制造术语词典</div>', unsafe_allow_html=True)
    st.markdown("搜索或浏览智能制造核心术语，点击查看详细解释。")
    
    col_search, col_cat, col_count = st.columns([2, 1, 1])
    with col_search:
        search = st.text_input("🔍 搜索术语", placeholder="输入术语名称，如 MLP、数字孪生...")
    with col_cat:
        categories = sorted(set([v["类别"] for v in TERMS.values()]))
        selected_cat = st.selectbox("按类别筛选", ["全部"] + categories)
    with col_count:
        st.metric("术语总数", f"{len(TERMS)} 个")
    
    # 过滤
    filtered = {}
    for term, info in TERMS.items():
        if search and search.lower() not in term.lower() and search.lower() not in info["全称"].lower():
            continue
        if selected_cat != "全部" and info["类别"] != selected_cat:
            continue
        filtered[term] = info
    
    if not filtered:
        st.warning("未找到匹配的术语，请尝试其他关键词。")
    else:
        st.caption(f"共找到 {len(filtered)} 个术语")
        for term, info in filtered.items():
            with st.expander(f"**{term}** — {info['全称']}", expanded=False):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**类别**  \n{info['类别']}")
                    st.markdown(f"**相关技术**  \n{info.get('相关技术', '—')}")
                with col2:
                    st.markdown(f"**📖 定义**  \n{info['定义']}")
                    st.markdown(f"**🏭 在智能制造中的应用**  \n{info['应用']}")

# ======================= 3. 数字孪生实验室 =======================
elif page == "💻 数字孪生实验室":
    st.markdown('<div class="main-header">💻 数字孪生实验室</div>', unsafe_allow_html=True)
    st.markdown("体验从 **物理实体 → 虚拟模型 → 实时交互** 的完整数字孪生闭环。")

    sub_page = st.radio(
        "选择孪生场景",
        ["🤖 3D机械臂 (运动学孪生)", 
         "📐 弹性末端 (动力学孪生)", 
         "📊 梁变形 (工程仿真孪生)",
         "⚙️ 切削加工 (制造过程孪生)"],
        horizontal=True
    )
    st.divider()
    
    # ================== 子模块1: 3D 机械臂 ==================
    if sub_page == "🤖 3D机械臂 (运动学孪生)":
        st.markdown('<div class="sub-header">🤖 六轴机械臂数字孪生 (运动学映射)</div>', unsafe_allow_html=True)
        st.caption("物理实体：关节角度 → 虚拟模型：3D机械臂 → 交互：滑块控制")
        
        # DH参数
        DH = [
            [0, np.pi/2, 400, 0],
            [600, 0, 0, -np.pi/2],
            [0, np.pi/2, 0, 0],
            [0, -np.pi/2, 600, 0],
            [0, np.pi/2, 0, 0],
            [0, 0, 200, 0]
        ]
        
        def forward(angles):
            pts = [np.array([0,0,0])]
            T = np.eye(4)
            for i in range(6):
                a, alpha, d, offset = DH[i]
                theta = angles[i] + offset
                ct, st = np.cos(theta), np.sin(theta)
                ca, sa = np.cos(alpha), np.sin(alpha)
                Ti = np.array([
                    [ct, -st*ca, st*sa, a*ct],
                    [st, ct*ca, -ct*sa, a*st],
                    [0, sa, ca, d],
                    [0,0,0,1]
                ])
                T = T @ Ti
                pts.append(T[:3,3])
            return np.array(pts).T[:,:-1], pts[-1]
        
        # 关节滑块
        st.markdown("**关节角度控制**")
        cols = st.columns(6)
        angles = []
        for i in range(6):
            with cols[i]:
                angles.append(st.slider(f"J{i+1}", -np.pi, np.pi, 0.0, step=0.05, key=f"dt_j{i}"))
        
        # 弹性参数
        st.markdown("**末端弹性参数**")
        col1, col2, col3 = st.columns(3)
        with col1:
            force = st.slider("末端外力 (N)", -500, 500, 0, key="dt_force")
        with col2:
            stiffness = st.slider("刚度 (N/m)", 500, 5000, 2000, key="dt_k")
        with col3:
            damping = st.slider("阻尼 (N·s/m)", 1, 50, 10, key="dt_damp")
        
        # 计算
        deformation = force / stiffness if stiffness != 0 else 0
        joints, tcp = forward(angles)
        actual_tcp = tcp + np.array([0, 0, deformation])
        
        # 3D绘图
        fig = go.Figure()
        for i in range(joints.shape[1]-1):
            x = [joints[0,i], joints[0,i+1]]
            y = [joints[1,i], joints[1,i+1]]
            z = [joints[2,i], joints[2,i+1]]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=6)))
        fig.add_trace(go.Scatter3d(x=joints[0], y=joints[1], z=joints[2], 
                                   mode='markers', marker=dict(size=6, color='black'), name='关节'))
        fig.add_trace(go.Scatter3d(x=[tcp[0]], y=[tcp[1]], z=[tcp[2]],
                                   mode='markers', marker=dict(size=8, color='green'), name='刚性末端'))
        fig.add_trace(go.Scatter3d(x=[actual_tcp[0]], y=[actual_tcp[1]], z=[actual_tcp[2]],
                                   mode='markers', marker=dict(size=10, color='red'), name='弹性末端'))
        fig.update_layout(height=500, 
                          scene=dict(xaxis_range=[-1000,1000], yaxis_range=[-1000,1000], zaxis_range=[0,1200],
                                     xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'))
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("弹性变形", f"{deformation*1000:.2f} mm")
        with col2:
            st.metric("末端位置", f"({actual_tcp[0]:.0f}, {actual_tcp[1]:.0f}, {actual_tcp[2]:.0f}) mm")
        
        st.info("💡 相关术语：**数字孪生**、**HCPS**、**边缘计算** — 物理关节角度实时映射到虚拟模型")

    # ================== 子模块2: 弹性变形 ==================
    elif sub_page == "📐 弹性末端 (动力学孪生)":
        st.markdown('<div class="sub-header">📐 弹性末端动态响应 (动力学孪生)</div>', unsafe_allow_html=True)
        st.caption("物理实体：外力 → 虚拟模型：弹簧-质量-阻尼系统 → 交互：参数调节")
        
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("刚度 k (N/m)", 500, 5000, 2000, key="dt2_k")
            c = st.slider("阻尼 c (N·s/m)", 5, 200, 30, key="dt2_c")
        with col2:
            m = st.slider("质量 m (kg)", 0.5, 10.0, 2.0, key="dt2_m")
            F0 = st.slider("阶跃力 (N)", 0, 500, 200, key="dt2_F")
        
        # 数值仿真
        dt = 0.005
        t = np.arange(0, 3.0, dt)
        x = np.zeros_like(t)
        v = 0.0
        
        for i in range(1, len(t)):
            a = (F0 - k*x[i-1] - c*v) / m
            v += a * dt
            x[i] = x[i-1] + v * dt
        
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(t, x*1000, 'r-', lw=2, label='弹性响应')
        ax.axhline(F0/k*1000, color='b', linestyle='--', label=f'稳态变形 {F0/k*1000:.2f} mm')
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("变形 (mm)")
        ax.set_title("弹性末端阶跃响应 (二阶系统)")
        ax.grid(True); ax.legend()
        st.pyplot(fig)
        
        # 性能指标
        col1, col2, col3 = st.columns(3)
        with col1:
            steady = F0/k*1000 if k>0 else 0
            st.metric("稳态变形", f"{steady:.2f} mm")
        with col2:
            overshoot = (np.max(x*1000) - steady)/steady*100 if steady>0 else 0
            st.metric("超调量", f"{max(0, overshoot):.1f}%")
        with col3:
            settle_idx = np.where(np.abs(x*1000 - steady) < 0.02*steady)[0]
            settle_time = t[settle_idx[0]] if len(settle_idx)>0 else 0
            st.metric("调节时间 (2%)", f"{settle_time:.2f} s")
        
        st.info("💡 相关术语：**MLP**（拟合响应曲线）、**LSTM**（预测变形趋势）、**数字孪生**（物理-虚拟映射）")

    # ================== 子模块3: 梁变形 ==================
    elif sub_page == "📊 梁变形 (工程仿真孪生)":
        st.markdown('<div class="sub-header">📊 简支梁变形计算 (工程仿真孪生)</div>', unsafe_allow_html=True)
        st.caption("物理实体：载荷 → 虚拟模型：材料力学公式 → 交互：参数调节")
        
        col1, col2 = st.columns(2)
        with col1:
            L = st.number_input("梁长 L (m)", 0.1, 10.0, 2.0, key="dt3_L")
            E = st.number_input("弹性模量 E (GPa)", 10, 300, 200, key="dt3_E") * 1e9
            b = st.number_input("宽度 b (mm)", 10, 200, 40, key="dt3_b") / 1000
            h = st.number_input("高度 h (mm)", 5, 100, 20, key="dt3_h") / 1000
        with col2:
            F = st.number_input("集中力 F (N)", 10, 10000, 1000, key="dt3_F")
            a = st.number_input("力的位置 a (m)", 0.0, L, L/2, key="dt3_a")
            n = st.number_input("分段数", 50, 500, 200, key="dt3_n")
        
        # 计算
        I = b * h**3 / 12
        x = np.linspace(0, L, int(n))
        y = np.zeros_like(x)
        
        for i, xi in enumerate(x):
            if xi <= a:
                y[i] = (F * xi * (L-a)) / (6 * E * I * L) * (L**2 - xi**2 - (L-a)**2)
            else:
                y[i] = (F * (L-xi) * a) / (6 * E * I * L) * (L**2 - (L-xi)**2 - a**2)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(x, y*1000, 'b-', lw=2, label='变形曲线')
        ax.axhline(0, color='k', linestyle='--')
        ax.set_xlabel("位置 (m)"); ax.set_ylabel("变形 (mm)")
        ax.set_title(f"简支梁变形 (最大变形 = {np.max(np.abs(y))*1000:.3f} mm)")
        ax.grid(True); ax.legend()
        st.pyplot(fig)
        
        # 关键结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最大变形", f"{np.max(np.abs(y))*1000:.3f} mm")
        with col2:
            max_pos = x[np.argmax(np.abs(y))]
            st.metric("最大变形位置", f"{max_pos:.2f} m")
        with col3:
            I_cm4 = I * 1e8
            st.metric("截面惯性矩", f"{I_cm4:.2f} cm⁴")
        
        st.info("💡 相关术语：**数字孪生**（梁的虚拟模型）、**MLP**（神经网络拟合变形曲线）")

    # ================== 子模块4: 切削加工 ==================
    else:
        st.markdown('<div class="sub-header">⚙️ 切削加工数字孪生 (力-变形-补偿闭环)</div>', unsafe_allow_html=True)
        st.caption("物理实体：切削参数 → 虚拟模型：力-变形预测 → 交互：实时补偿")
        
        col_left, col_mid, col_right = st.columns([1.2, 2, 1])

        with col_left:
            st.markdown("**🔧 加工参数**")
            st.caption("工件参数")
            L = st.number_input("长度 L (mm)", 50, 300, 120, step=10, key="cut_L")
            W = st.number_input("宽度 W (mm)", 20, 100, 40, step=5, key="cut_W")
            t = st.number_input("厚度 t (mm)", 1, 10, 3, step=0.5, key="cut_t")
            
            st.caption("切削参数")
            F = st.slider("切削力 F (N)", 100, 1500, 500, step=50, key="cut_F")
            tool_pos = st.slider("刀具位置 (mm)", 0, L, L//2, step=5, key="cut_pos")
            
            material = st.selectbox("材料", ["铝合金 (E=70GPa)", "钢材 (E=200GPa)", "钛合金 (E=110GPa)"], key="cut_mat")
            E_map = {"铝合金 (E=70GPa)": 70e9, "钢材 (E=200GPa)": 200e9, "钛合金 (E=110GPa)": 110e9}
            E = E_map[material]
            
            compensate = st.checkbox("✅ 启用刀路补偿", value=True, key="cut_comp")
            
            st.markdown("---")
            st.caption("💡 **孪生逻辑**：切削力 → 预测变形 → 生成补偿刀路")

        with col_mid:
            st.markdown("**📐 变形预测与补偿可视化**")
            
            L_m = L / 1000
            W_m = W / 1000
            t_m = t / 1000
            I = W_m * t_m**3 / 12
            
            n_points = 100
            x = np.linspace(0, L_m, n_points)
            y_def = np.zeros_like(x)
            
            a_m = tool_pos / 1000
            for i, xi in enumerate(x):
                if xi <= a_m:
                    y_def[i] = (F * xi * (L_m - a_m)) / (6 * E * I * L_m) * (L_m**2 - xi**2 - (L_m - a_m)**2)
                else:
                    y_def[i] = (F * (L_m - xi) * a_m) / (6 * E * I * L_m) * (L_m**2 - (L_m - xi)**2 - a_m**2)
            
            y_comp = -y_def if compensate else np.zeros_like(y_def)
            max_def = np.max(np.abs(y_def)) * 1000
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x*1000, y=y_def*1000,
                mode='lines',
                name='原始变形 (向下)',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=x*1000, y=y_comp*1000,
                mode='lines',
                name='补偿刀路' if compensate else '未启用补偿',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, L], y=[0, 0],
                mode='lines',
                name='理想表面',
                line=dict(color='green', width=1, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=[tool_pos], y=[0],
                mode='markers',
                name='刀具位置',
                marker=dict(color='orange', size=12, symbol='triangle-down')
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title="工件长度 (mm)",
                yaxis_title="变形量 (mm)",
                yaxis=dict(autorange=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            tool_idx = int(tool_pos / L * n_points)
            if tool_idx < n_points:
                def_at_tool = y_def[tool_idx] * 1000
                comp_at_tool = y_comp[tool_idx] * 1000
                st.caption(f"📍 刀具位置 {tool_pos}mm 处：变形 {def_at_tool:.3f}mm，补偿 {comp_at_tool:.3f}mm")

        with col_right:
            st.markdown("**📊 孪生结果**")
            
            st.metric("最大变形", f"{max_def:.3f} mm", delta="需补偿" if max_def > 0.05 else "可接受")
            
            if compensate:
                residual = max_def * 0.05
                st.metric("补偿后残余误差", f"{residual:.4f} mm", delta="↓ 减小95%", delta_color="inverse")
            else:
                st.metric("无补偿误差", f"{max_def:.3f} mm", delta="⚠️ 超差风险", delta_color="inverse")
            
            if max_def < 0.05:
                st.success("✅ 加工精度满足要求")
            elif max_def < 0.15:
                st.warning("⚡ 建议启用补偿")
            else:
                st.error("🚨 变形超差，必须启用补偿")
            
            st.markdown("---")
            st.caption("**补偿效果对比**")
            
            fig_bar, ax_bar = plt.subplots(figsize=(4, 2.5))
            labels = ['无补偿', '有补偿']
            values = [max_def, max_def * 0.05 if compensate else max_def]
            bars = ax_bar.bar(labels, values, color=['#FF6B6B', '#51CF66'])
            ax_bar.set_ylabel('最大误差 (mm)')
            ax_bar.set_title('补偿效果对比')
            ax_bar.grid(axis='y', linestyle='--', alpha=0.3)
            for bar, val in zip(bars, values):
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            st.pyplot(fig_bar)
            
            st.markdown("---")
            st.caption("**🔄 孪生闭环**")
            st.code("""
切削力 → 变形预测 → 刀路补偿 → 加工执行 → 反馈优化
            """, language="text")

# ======================= 4. 知识问答 =======================
elif page == "❓ 知识问答":
    st.markdown('<div class="main-header">❓ 智能制造知识问答</div>', unsafe_allow_html=True)
    st.markdown("选择问题，点击按钮查看答案，巩固课程知识点。")
    
    qa = {
        "智能制造的核心是什么？": "数字化、网络化、智能化与制造技术的深度融合。",
        "HCPS 是什么？": "人-信息-物理系统 (Human-Cyber-Physical Systems)。",
        "工业机器人的三大核心部件？": "减速器、伺服电机、控制器。",
        "数字孪生的核心价值？": "物理实体与虚拟模型的双向映射与实时交互。",
        "PLC 是什么？": "可编程逻辑控制器，工业现场的控制核心。",
        "MES 的作用？": "制造执行系统，连接计划层与设备层。",
        "MLP 在智能制造中的作用？": "用于工艺参数优化、设备故障预测、质量分类等。",
        "边缘计算的优势？": "低延迟、数据不出厂、高可靠性。",
        "工业互联网的核心？": "设备联网、数据采集、跨系统协同。",
        "什么是工业大模型？": "面向工业领域的大规模预训练模型，用于智能排产、知识问答等。",
        "SCADA 与 PLC 的关系？": "PLC负责现场控制，SCADA负责集中监控和调度。",
        "数字主线与数字孪生的区别？": "数字主线是数据流，数字孪生是模型+数据+交互的闭环系统。",
        "具身智能的定义？": "智能体通过物理实体与环境交互，实现感知-决策-行动一体化。",
        "BOM 是什么？": "物料清单 (Bill of Materials)，是PLM/MES/ERP协同的数据核心。",
        "DCS 的应用场景？": "流程工业（化工、炼油、电力）的大型控制系统。",
        "CNN 在制造中的应用？": "表面缺陷检测、工件识别分类、视觉引导机器人。",
        "工业智能体的特点？": "感知环境、规划任务、调用工具、自主决策执行。",
        "数字孪生系统的构成？": "物理实体、虚拟模型、数据交互、服务应用。",
        "边缘计算与云计算的区别？": "边缘计算靠近数据源，实时性好；云计算中心化，算力更强。",
        "HCPS 的三要素？": "人 (Human)、信息 (Cyber)、物理 (Physical) 系统。"
    }
    
    col_q, col_a = st.columns([1, 1])
    with col_q:
        question = st.selectbox("选择问题", list(qa.keys()))
        if st.button("显示答案", use_container_width=True):
            st.session_state['answer'] = qa[question]
    with col_a:
        if 'answer' in st.session_state:
            st.success(f"✅ {st.session_state['answer']}")
        else:
            st.info("💡 点击左侧「显示答案」按钮查看")
    
    st.markdown("---")
    st.caption("📖 如需了解更多术语，请前往「术语查询」模块。")
    
    with st.expander("📊 学习进度自测", expanded=False):
        st.markdown("**请自我评估对以下知识点的掌握程度：**")
        topics = [
            "智能制造基本概念",
            "HCPS 人-信息-物理系统",
            "工业机器人原理与应用",
            "PLM/MES/ERP 工业软件",
            "数字孪生技术",
            "工业互联网架构",
        ]
        for topic in topics:
            st.slider(f"✓ {topic}", 0, 10, 5, key=f"quiz_{topic}")