import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Line, Pie
from st_aggrid import AgGrid
from streamlit_echarts import st_pyecharts
from tqdm import tqdm
from copy import copy
from common_utils import load_data,get_time_keys,smooth,get_trend_to_show
from inference import get_es_data, show_trend, sort_data, analysis_info, get_co_occurrence_info, get_co_matrix

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


if ('any_target_map_dict' not in st.session_state):
    candidate = load_data('../data/search_candidate.pkl')
    st.session_state.all_indication = candidate['indication']
    st.session_state.all_target = candidate['target']
    st.session_state.all_technology = candidate['technology']
    st.session_state.all_cls_type = {'黑名单', '非临床', '临床', '其他', '遗传学', '结构'}
    st.session_state.any_target_map_dict = {
        '随意': 'None',
        '是': 'True',
        '否': 'False'
    }

start_time = str(st.sidebar.date_input(
    "文章范围开始时间(包含该时间)",
    datetime.date(2020, 1, 1),
    min_value=datetime.date(1990, 1, 1),
    max_value=datetime.date(2023, 12, 31)))
if 'start_time' not in st.session_state:
    st.session_state.start_time = start_time

end_time = str(st.sidebar.date_input(
    "文章范围结束时间(包含该时间)",
    datetime.date(2023, 3, 31),
    min_value=datetime.date(1990, 1, 1),
    max_value=datetime.date(2023, 12, 31)
))
if 'end_time' not in st.session_state:
    st.session_state.end_time = end_time


time_interval = st.sidebar.number_input(
    label="统计时间的间隔（月）",
    value=1,
    min_value=1,
    max_value=36
)
if 'time_interval' not in st.session_state:
    st.session_state.time_interval = time_interval


indication = st.sidebar.multiselect(
    '输入或选择多个疾病(默认选择所有)',
    st.session_state.all_indication,
    ['乳腺癌']
)  # list
if 'indication' not in st.session_state:
    st.session_state.indication = indication

any_target = st.session_state.any_target_map_dict[st.sidebar.selectbox(
    '文章中是否含有任意靶点',
    ['随意', '是', '否'],
)]

if 'any_target' not in st.session_state:
    st.session_state.any_target = any_target

target = st.sidebar.multiselect(
    '输入或选择多个靶点(默认选择所有)',
    st.session_state.all_target,
    ['HER2']
)

if 'target' not in st.session_state:
    st.session_state.target = target

technology = st.sidebar.multiselect(
    '输入或选择多个技术标签(默认选择所有)',
    st.session_state.all_technology,
    ['单抗']
)
if 'technology' not in st.session_state:
    st.session_state.technology = technology

cls_type = st.sidebar.multiselect(
    '输入或选择多个文章类型(默认选择所有)',
    st.session_state.all_cls_type,
    []
)
if 'cls_type' not in st.session_state:
    st.session_state.cls_type = cls_type

max_data_length = 100000


if start_time >= end_time:
    st.warning(f'时间范围设置错误，结束时间应该大于开始时间！')


if ('es_data' not in st.session_state) or (start_time != st.session_state.start_time) or \
        (end_time != st.session_state.end_time) or (indication != st.session_state.indication) or \
        (any_target != st.session_state.any_target) or (target != st.session_state.target) \
        or (technology != st.session_state.technology) or (cls_type != st.session_state.cls_type):
    st.session_state.es_data, st.session_state.es_state = get_es_data(
        time_limit_range=(start_time, end_time),
        indication=indication,
        target=target,
        technology=technology,
        cls_type=cls_type,
        any_target=any_target,  # None，False，Ture
        max_data_length=max_data_length)
    st.session_state.start_time = start_time
    st.session_state.end_time = end_time
    st.session_state.indication = indication
    st.session_state.any_target = any_target
    st.session_state.target = target
    st.session_state.technology = technology
    st.session_state.cls_type = cls_type
    st.session_state.time_interval = time_interval

    st.session_state.score_fields = ['被引用数', '期刊影响因子', '月均引用']

    st.session_state.score_fields_mapping = {
        '被引用数': 'citedby_calculated_count',
        '期刊影响因子': 'journal_impact_factor_value',
        '月均引用': 'monthly_citation_count',
    }

    st.session_state.base_doc_trend = show_trend(st.session_state.es_data, field=None)  # dict()


if st.session_state.es_state != 'right':
    st.warning(f'根据条件搜索出来的文档总数大于{max_data_length}篇，这边只呈现最近时间的{max_data_length}篇文档，请缩小过滤条件！')
else:
    st.success(f'搜索完成，满足搜索条件的文档数有{len(st.session_state.es_data)}篇！')

all_time_keys = get_time_keys(st.session_state.start_time[0:-3],
                                  st.session_state.end_time[0:-3],
                                  time_interval)


smooth_para = st.slider('趋势平滑系数（可通过调节该参数更方便地看出图像的变化趋势）:', 1, 50, 1)
if len(st.session_state.es_data) > 0:
    doc_trend_key,doc_trend_value = get_trend_to_show(all_time_keys, st.session_state.base_doc_trend)
    if smooth_para!=1:
        doc_trend_key,doc_trend_value = smooth( doc_trend_key,doc_trend_value, smooth_para,remove_bound=True)
        if not doc_trend_key:
            st.warning('平滑范围太大!请缩小范围！')
    trend_bar = (
        Line()
            .add_xaxis(doc_trend_key)
            .add_yaxis(series_name='文档数趋势图', y_axis=[round(i,2) for i in doc_trend_value])
            .set_global_opts(title_opts=opts.TitleOpts(title="文档数随时间变化图"),
                             toolbox_opts=opts.ToolboxOpts())
    )
    st_pyecharts(trend_bar)
else:
    st.warning('搜索出 0 篇文档，请重新设置过滤条件！')

add_field, info_tab, analysis_tab, graph_tab = st.tabs(["自定义文档得分", "信息展示", "数据分析", "图谱展示"])

with add_field:
    st.subheader('1. 自定义文档得分:')
    st.info("说明：可以自定义每个文档的得分计算公式并作为字段加入到数据中进行呈现，以下（1）需要自定义一个字段名称，（2）需要自定义一个得分"
            "计算公式，现支持对citedby_calculated_count（引用数）、monthly_citation_count（月均被引）、journal_impact_factor_"
            "value（影响因子）三个字段进行加减乘除（+=*/）的得分计算。 ")
    st.info("比如例子： 【 doc['citedby_calculated_count']*2 + doc['monthly_"
            "citation_count']*5 + doc['journal_impact_factor_value']*4 】 表示：对引用数乘以2，加上对月均被引乘以5，加上对影响因子乘以4 所获得的得分")
    score_field = st.text_input(label='(1) 字段名称:', value='my_score_field')
    score_formula = st.text_area(label='(2) 得分计算公式:',
                                 value="doc['citedby_calculated_count']*2 + doc['monthly_citation_count']*5 + doc['journal_impact_factor_value']*4")

    if st.button('添加自定义得分'):
        if score_field not in st.session_state.score_fields:
            exec_formula = f"doc['{score_field}'] = " + score_formula
            deal_with_bar = st.progress(0)
            count = 1
            for doc in tqdm(st.session_state.es_data, desc='add new field...'):
                deal_with_bar.progress(count / len(st.session_state.es_data))
                exec(exec_formula)
                count += 1
            st.session_state.score_fields.append(score_field)
            st.session_state.score_fields_mapping[score_field] = score_field
            st.success(f'字段：{score_field}，得分添加完成！')
        else:
            st.error('该字段已经被使用！')

    st.subheader('2. 统计文档得分随时间的变化')
    st.info('说明：指定字段，统计时间段内所有文档中该字段的得分总和!')
    select_score_field = st.selectbox(
        '得分统计字段：',
        st.session_state.score_fields
    )

    score_field_trend = show_trend(st.session_state.es_data,
                                   field=st.session_state.score_fields_mapping[select_score_field])

    score_field_trend_key,score_field_trend_keys_value = get_trend_to_show(all_time_keys,score_field_trend)

    if smooth_para != 1:
        score_field_trend_key,score_field_trend_keys_value = smooth(score_field_trend_key,
                                                                    score_field_trend_keys_value, smooth_para,
                                                                    remove_bound=True)
        if not score_field_trend_key:
            st.warning('平滑范围太大!请缩小范围！')

    trend_bar2 = (
        Line()
            .add_xaxis(score_field_trend_key)
            .add_yaxis(series_name='趋势图', y_axis=[round(i,2) for i in score_field_trend_keys_value])
            .set_global_opts(title_opts=opts.TitleOpts(title="文档统计得分随时间变化图"),
                             toolbox_opts=opts.ToolboxOpts())
    )
    st_pyecharts(trend_bar2)

with info_tab:
    st.info('说明：展示部分文档基本信息，可以排序文档以及设置显示文档的字段和数量')
    sorted_fields = copy(st.session_state.score_fields)
    sorted_fields_mapping = copy(st.session_state.score_fields_mapping)
    sorted_fields.append('发表时间')
    sorted_fields_mapping['发表时间'] = 'pub_date'

    sorted_field = st.selectbox(
        '排序字段：',
        sorted_fields
    )

    sorted_order = st.selectbox(
        '顺序或倒序：',
        ['倒序', '顺序']
    )
    sorted_order_mapping = {
        '顺序': 'asc',
        '倒序': 'desc'
    }

    top_k = st.selectbox(
        '选择top-k文章展示：',
        (3, 1, 10, 20, 30,)
    )

    show_field_mapping = {
        '标题': 'title',
        '发表时间': 'pub_date',
        '被引用数': 'citedby_calculated_count',
        'pubmed_id': 'pmid',
        '期刊影响因子': 'journal_impact_factor_value',
        '文章分类': 'cls_type',
        '所有可能涉及的实体': 'entry',
        '实体间的粗略关系': 'relation',
        '月均引用': 'monthly_citation_count'
    }
    show_field_mapping.update(st.session_state.score_fields_mapping)
    show_field = st.multiselect(
        '要显示的字段信息：',
        list(show_field_mapping.keys()),
        ['pubmed_id', '标题', '所有可能涉及的实体', '实体间的粗略关系']
    )
    if sorted_field not in show_field:
        show_field.append(sorted_field)

    sub_data = sort_data(st.session_state.es_data, sorted_field=sorted_fields_mapping[sorted_field],
                         order=sorted_order_mapping[sorted_order])[0:top_k]

    info = [{j: i[show_field_mapping[j]] for j in show_field} for i in sub_data]
    st.json(info)

with analysis_tab:
    st.info('说明：点击按钮则进行一些简单的数据分析!')

    if st.button('进行数据分析'):
        analysis_res = analysis_info(st.session_state.es_data, top_k=20)

        st.info(f'根据过滤条件筛选出 {analysis_res["article_num"]} 篇文档!')
        if analysis_res["article_num"] > 0:
            st.subheader('文档分类占比')
            cls_pie = (
                Pie()
                    .add('文档分类占比饼图', analysis_res['cls_type_ana'])
            )
            st_pyecharts(cls_pie)

            st.subheader('占比最多的top-20个疾病/靶点/技术标签词条')
            di_ana, tar_ana, tech_ana = st.columns(3)
            with di_ana:
                st.subheader("疾病")
                di_df = pd.DataFrame(data=analysis_res['indication_type_ana'], columns=['疾病', '文档数'])
                AgGrid(di_df, fit_columns_on_grid_load=True)

            with tar_ana:
                st.subheader("靶点")
                tar_df = pd.DataFrame(data=analysis_res['target_type_ana'], columns=['靶点', '文档数'])
                AgGrid(tar_df, fit_columns_on_grid_load=True)

            with tech_ana:
                st.subheader("技术标签")
                tech_df = pd.DataFrame(data=analysis_res['technology_type_ana'], columns=['技术标签', '文档数'])
                AgGrid(tech_df, fit_columns_on_grid_load=True)


def create_graph(two_relations, three_relations, max_relation_num=100):
    two_relations = sorted(two_relations.items(), key=lambda x: x[-1], reverse=True)[0:max_relation_num]
    three_relations = sorted(three_relations.items(), key=lambda x: x[-1], reverse=True)[0:int(max_relation_num / 4)]
    g = nx.Graph()

    for i in two_relations:
        node1 = i[0][0]
        node2 = i[0][1]
        if node1 not in g:
            g.add_node(node1, type=i[0][2])
        if node2 not in g:
            g.add_node(node2, type=i[0][3])
        g.add_edge(node1, node2, weight=i[-1])
    for i in three_relations:
        node1 = i[0][0]
        node2 = i[0][1]
        node3 = i[0][2]
        if node1 not in g:
            g.add_node(node1, type='indication')
        if node2 not in g:
            g.add_node(node2, type='target')
        if node3 not in g:
            g.add_node(node3, type='technology')
        if (node1, node2) not in g.edges:
            g.add_edge(node1, node2, weight=1)
        if (node1, node3) not in g.edges:
            g.add_edge(node1, node3, weight=1)
        if (node2, node3) not in g.edges:
            g.add_edge(node2, node3, weight=1)

    pos = nx.spring_layout(g)
    indication_node = [i for i in g.nodes() if g.nodes[i]['type'] == 'indication']
    target_node = [i for i in g.nodes() if g.nodes[i]['type'] == 'target']
    technology_node = [i for i in g.nodes() if g.nodes[i]['type'] == 'technology']

    nx.draw_networkx_nodes(g, pos=pos, nodelist=indication_node, node_color='r', label='indication')
    nx.draw_networkx_nodes(g, pos=pos, nodelist=target_node, node_color='y', label='target')
    nx.draw_networkx_nodes(g, pos=pos, nodelist=technology_node, node_color='b', label='technology')

    nx.draw_networkx_labels(g, pos=pos, labels={i: i for i in indication_node},
                            font_size=4)
    nx.draw_networkx_labels(g, pos=pos, labels={i: i for i in target_node},
                             font_size=4)
    nx.draw_networkx_labels(g, pos=pos, labels={i: i for i in technology_node},
                             font_size=4)

    nx.draw_networkx_edges(g, pos=pos, edgelist=g.edges)
    return g


with graph_tab:
    st.info('说明：点击按钮则进行一些简单的实体词条关系展现，注意：这边的关系只是表明这两个词条出现在同一段落中!')
    if st.button('图谱共现分析'):
        two_relations, three_relations = get_co_occurrence_info(st.session_state.es_data)

        if len(two_relations) > 0:
            st.subheader('部分图谱信息展示（其中红色表示疾病，黄色表示靶点，蓝色表示技术）')
            fig, ax = plt.subplots()
            g = create_graph(two_relations, three_relations, max_relation_num=80)
            st.pyplot(fig)

            indication_target, indication_technology, target_technology = get_co_matrix(
                co_occurrence_info=two_relations,
                max_size=6)
            if len(indication_target) > 0:
                indication_target.reset_index(level=0, inplace=True)
                indication_target.rename(columns={'index': '疾病-靶点'}, inplace=True)
                st.subheader('疾病-靶点部分共现信息：')
                AgGrid(indication_target, fit_columns_on_grid_load=True, sideBar=True)

            if len(indication_technology) > 0:
                indication_technology.reset_index(level=0, inplace=True)
                indication_technology.rename(columns={'index': '疾病-技术标签'}, inplace=True)
                st.subheader('疾病-技术标签部分共现信息：')
                AgGrid(indication_technology, fit_columns_on_grid_load=True, sideBar=True)

            if len(target_technology) > 0:
                target_technology.reset_index(level=0, inplace=True)
                target_technology.rename(columns={'index': '靶点-技术标签'}, inplace=True)
                st.subheader('靶点-技术标签部分共现信息：')
                AgGrid(target_technology, fit_columns_on_grid_load=True, sideBar=True)

        if len(three_relations) > 0:
            three_relations = sorted(three_relations.items(), key=lambda x: x[-1], reverse=True)[0:10]
            three_relations = [{'疾病-靶点-技术标签': '-'.join(i), '文档数': j} for i, j in three_relations]

            three_relations = pd.DataFrame(three_relations)
            st.subheader('疾病-靶点-技术标签 部分共现信息：')
            AgGrid(three_relations, fit_columns_on_grid_load=True, sideBar=True)
