try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

# ==========================================
# 1. 画布与全局样式初始化
# ==========================================
dot = Digraph(name='VASR_Module', format='pdf')
dot.attr(rankdir='LR', splines='ortho', nodesep='0.6', ranksep='0.8')
dot.attr('node', shape='box', style='filled', fontname='Times New Roman', fontsize='14', margin='0.2')
dot.attr('edge', fontname='Times New Roman', fontsize='12')

# 定义不同分支的颜色主题
COLOR_SEMANTIC = '#e2f0d9'  # 浅绿色 (属性语义)
COLOR_VISUAL = '#deebf7'  # 浅蓝色 (全局视觉)
COLOR_MODULE = '#f2f2f2'  # 浅灰色 (计算模块)
COLOR_OP = '#fff2cc'  # 浅黄色 (操作符)

# ==========================================
# 步骤 1：布置输入端
# ==========================================
dot.node('Va', 'Attribute Semantics (Va)\n[B, N_attr, 768]', fillcolor=COLOR_SEMANTIC, color='#548235', penwidth='2')
dot.node('fg', 'Global Visual Feature (fg)\n[B, 768]', fillcolor=COLOR_VISUAL, color='#2e75b6', penwidth='2')

# ==========================================
# 步骤 2：绘制“语义自注意力”分支
# ==========================================
dot.node('MHSA', 'Multi-Head\nSelf-Attention', fillcolor=COLOR_MODULE)
dot.node('Add1', '+', shape='circle', fillcolor=COLOR_OP, width='0.4', fixedsize='true')
dot.node('LN1', 'LayerNorm', fillcolor=COLOR_MODULE)
dot.node('V_hat', 'Preliminary\nSemantic (V_hat)', fillcolor=COLOR_SEMANTIC, style='rounded,filled')

dot.edge('Va', 'MHSA')
dot.edge('MHSA', 'Add1')
# 残差连接
dot.edge('Va', 'Add1', label=' Residual', constraint='false', style='dashed')
dot.edge('Add1', 'LN1')
dot.edge('LN1', 'V_hat')

# ==========================================
# 步骤 3：绘制“视觉门控”分支
# ==========================================
with dot.subgraph(name='cluster_Gate') as c:
    c.attr(label='Visual Gate Mechanism', fontname='Times New Roman', style='dashed', color='gray')
    c.node('L1', 'Linear (W2)', fillcolor=COLOR_MODULE)
    c.node('ReLU', 'ReLU', fillcolor=COLOR_MODULE)
    c.node('L2', 'Linear (W1)', fillcolor=COLOR_MODULE)
    c.node('Sigmoid', 'Sigmoid', fillcolor=COLOR_MODULE)

    c.edge('L1', 'ReLU')
    c.edge('ReLU', 'L2')
    c.edge('L2', 'Sigmoid')

dot.edge('fg', 'L1')
dot.node('Gate', 'Gate (g)', fillcolor=COLOR_VISUAL, style='rounded,filled')
dot.edge('Sigmoid', 'Gate')

# ==========================================
# 步骤 4：核心交互（视觉引导语义）
# ==========================================
dot.node('Mul', 'X', shape='circle', fillcolor=COLOR_OP, width='0.4', fixedsize='true', label='⊙')
dot.edge('V_hat', 'Mul')
dot.edge('Gate', 'Mul')

# ==========================================
# 步骤 5：绘制“特征精炼”分支
# ==========================================
dot.node('FFN', 'Feed-Forward\nNetwork\n(Linear->ReLU->Linear)', fillcolor=COLOR_MODULE)
dot.node('Add2', '+', shape='circle', fillcolor=COLOR_OP, width='0.4', fixedsize='true')
dot.node('LN2', 'LayerNorm', fillcolor=COLOR_MODULE)
dot.node('V_tilde', 'Visual-Aware\nSemantic Features (V_tilde)\n[B, N_attr, 768]', fillcolor=COLOR_SEMANTIC, color='#548235', penwidth='2')

dot.edge('Mul', 'FFN')
dot.edge('FFN', 'Add2')
# 第二次残差连接
dot.edge('Mul', 'Add2', label=' Residual', constraint='false', style='dashed')
dot.edge('Add2', 'LN2')
dot.edge('LN2', 'V_tilde')

# ==========================================
# 生成结构图
# ==========================================
dot.render('VASR_Module_Architecture', view=True)
print("✅ VASR 模块架构图已成功生成为 PDF！")