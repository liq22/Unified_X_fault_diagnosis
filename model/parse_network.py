
import networkx as nx
import matplotlib.pyplot as plt
import torch

# prune network

def draw_signal_processing_layer(G,layer, layer_idx,input_nodes): # 
    # 
    
    # 获取weight_connection的权重
    weight = layer.weight_connection.weight.detach().numpy() # 获取权重
    module_num = layer.module_num # 信号处理模块的数量
    
    in_channel = weight.shape[1] # 输入通道
    out_channel = weight.shape[0] # 输出通道
    
    num_per_module = out_channel // module_num # 每个模块的输入通道数量
    # module_name_list = layer.signal_processing_modules.values() # 信号处理模块的名称
    
    # 计算位置和添加节点

    
    output_nodes = [f'$x^{layer_idx + 1}_{j}$' for j in range(out_channel)]
    # G.add_nodes_from(input_nodes, layer='input')
    G.add_nodes_from(output_nodes, layer='output') # 不需要隐藏掉了
    
    # # 添加边
    # for i, input_node in enumerate(input_nodes):
    #     for j, output_node in enumerate(hidden_nodes):
    #         # 根据权重调整边的属性
    #         G.add_edge(input_node, output_node, weight=abs(weight[j, i]))
    
    # 添加模块节点
    module_nodes = []
    for idx, module in enumerate(layer.signal_processing_modules.values(), 1):
        for i in range(num_per_module):
            module_name = f'{module.name}_{idx}'
            module_nodes.append(module_name)
    G.add_nodes_from(module_nodes, layer='module')
    # 添加边
    for i, input_node in enumerate(input_nodes):
        for j, module_node in enumerate(module_nodes):
            # 根据权重调整边的属性   
            G.add_edge(input_node, module_node,weight=weight[j, i])
            G.add_edge(module_node, output_nodes[j])
    
    
    # 如果存在skip_connection，则添加跳跃连接
    if hasattr(layer, 'skip_connection'):
        skip_weight = layer.skip_connection.weight.detach().numpy()
        
        # output_nodes = [f'$O_{j}$' for j in range(out_channel)] # 输出节点 
        for i, input_node in enumerate(input_nodes):
            for j, output_node in enumerate(output_nodes):
                G.add_edge(input_node, output_node, weight = skip_weight[j, i], skip=True)
    return G, output_nodes

def draw_signal_processing_layers(model, input):
    G = nx.Graph()
    input_nodes = [f'$x^0_{j}$' for j in range(input.shape[2])]
    for idx, layer in enumerate(model.signal_processing_layers):
        G, input_nodes = draw_signal_processing_layer(G,layer, idx, input_nodes)    
    
    # 使用networkx绘制图形
    pos = nx.spring_layout(G)  # 可以根据需要选择不同的布局
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, edges=G.edges(), width=weights)
    plt.title(f'Signal Processing Layers')
    plt.show()
    return G, input_nodes

def draw_feature_extractor_layer(G,layer,input_nodes):
    weight = layer.weight_connection.weight.detach().numpy() # 获取权重
    in_channel = weight.shape[1] # 输入通道
    out_channel = weight.shape[0] # 输出通道
    output_nodes = [f'$f_{j}$' for j in range(out_channel)]
    feature_nodes = [f'$\\tau_{j}$' for j in range(out_channel)]
    G.add_nodes_from(output_nodes, layer='output')
    G.add_nodes_from(feature_nodes, layer='feature')
    for i, input_node in enumerate(input_nodes):
        for j, output_node in enumerate(output_nodes):
            G.add_edge(input_node, feature_nodes[j], weight=weight[j, i])
            G.add_edge(feature_nodes[j],output_node )
    return G, output_nodes

def draw_classifier_layer(G,layer,input_nodes):
    weight = layer.weight_connection.weight.detach().numpy() # 获取权重
    in_channel = weight.shape[1] # 输入通道
    out_channel = weight.shape[0] # 输出通道
    output_nodes = [f'$y_{j}$' for j in range(out_channel)]
    G.add_nodes_from(output_nodes, layer='output')
    for i, input_node in enumerate(input_nodes):
        for j, output_node in enumerate(output_nodes):
            G.add_edge(input_node, output_node, weight=weight[j, i])
    return G, output_nodes
    
# 创建模型实例和dummy input
# model = Transparent_Signal_Processing_Network(...)
# input_tensor = torch.randn((1, input_channels, input_length))  # 假设的输入

# 调用绘图函数
# draw_network(model, input_tensor)

if __name__ == '__main__':
    # 示例参数
    from configs.config import args
    from configs.config import signal_processing_modules,feature_extractor_modules
    from TSPN import Transparent_Signal_Processing_Network
    import torch
    
    net = Transparent_Signal_Processing_Network(signal_processing_modules,feature_extractor_modules, args)
    x = torch.randn(2, 4096, 2)
    y = net(x)
    print(y.shape)
    
    signal_processing_modules = [{'type': 'conv', 'out_channels': 64}, {'type': 'conv', 'out_channels': 128}]
    feature_extractor_modules = [{'type': 'conv', 'out_channels': 64}, {'type': 'pool'}]
    args = {
        'in_channels': 1,
        'out_channels': 64,
        'scale': 2,
        'skip_connection': True,
        'num_classes': 10
    }

    # 调用函数进行测试
    # draw_network(net, x)
