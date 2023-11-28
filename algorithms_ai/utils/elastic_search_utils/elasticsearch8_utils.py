"""
elasticsearch8-utils

获取es数据 es_batch_search_data
上传es数据 es_bulk_upload_data(多条)，es_iter_upload_data（单挑）
删除es数据 delete_by_query，delete_by_id
esquery数：es_count
别名：get_alias
获取setting：get_index_setting
清空es：clear_index
检查es是否存在index：check_es_exist_index
"""

from elasticsearch.helpers import bulk
from loguru import logger
from tqdm import tqdm


def delete_by_id(es_client, index_name, id: str):
    """Delete one document by id.

    Args:
        id: Id of doc to delete.
    """
    es_client.es.delete(index=index_name, id=id, refresh=True)


def delete_by_query(es_client, index_name, query_field_name: str, query_field_value):
    """Delete documents by certain field name.

    Delete all documents if selected field value equals query_field_value.

    Args:
        query_field_name: Name of the field.
        query_field_value: Value of the field.
    """
    query_doc = {
        "query": {
            "term": {
                query_field_name: query_field_value
            },
        },
    }
    es_client.delete_by_query(index=index_name, body=query_doc, refresh=True)


def es_batch_search_data(es_client, index_name, batch_size=1000, query=None, source=None, fields=None, sort=None,
                         track_total_hits=True):
    """
    从es中取所有数据，以batch的形式取search结果
    :param es_client: es服务
    :param index_name: index名称
    :param batch_size: batch大小
    :param query: query是什么
    :param source: 保留的字段
    :param fields: 保留的fields
    :param sort: 排序的配置
    :param track_total_hits: 是否取所有
    :return: 迭代的数据
    """
    check_es_exist_index(es_client, index_name)

    if not query:
        query = {
            "match_all": {}
        }
    if not sort:
        sort = [{'_id': "asc"}]

    data = []
    res = es_client.search(index=index_name, query=query, size=batch_size, sort=sort, source=source, fields=fields,
                           track_total_hits=track_total_hits)
    append_data = data.append
    for r in res['hits']['hits']:
        doc = r['_source']
        doc["_id"] = r['_id']
        if fields:
            doc.update(r.get('fields', dict()))
        append_data(doc)
    yield data

    if res["hits"]["hits"] and 'sort' in res["hits"]["hits"][-1] and res["hits"]["hits"][-1]["sort"]:
        bookmark = res["hits"]["hits"][-1]["sort"]

        if res['hits'].get('total', {}).get('relation', '') == 'eq':
            total = res['hits'].get('total', {}).get('value')
        else:
            total = es_count(es_client, index_name, query=query)

        logger.info(f'Total count:{total}')

        for _ in tqdm(range(total // batch_size), desc=f'get es-data, batch size={batch_size}'):
            data = []
            append_data = data.append
            res = es_client.search(index=index_name, query=query, size=batch_size, sort=sort, source=source,
                                   fields=fields, track_total_hits=track_total_hits, search_after=bookmark)
            if not res["hits"]["hits"]:
                break
            for r in res['hits']['hits']:
                doc = r['_source']
                doc["_id"] = r['_id']
                if fields:
                    doc.update(r.get('fields', dict()))
                append_data(doc)
            yield data

            if res["hits"]["hits"] and 'sort' in res["hits"]["hits"][-1] and res["hits"]["hits"][-1]["sort"]:
                bookmark = [res["hits"]["hits"][-1]["sort"][0]]
                return_size = res["hits"]["total"]["value"]
                if return_size < batch_size:
                    break
            else:
                break
    logger.success(f'Search es completely!')


def es_bulk_upload_data(es_client, index_name, data, action='update', id_key='_id'):
    """
    传入一批数据给es，如果需要batch上传数据，要写在外面
    :param es_client:
    :param index_name:
    :param data: 数据是一个list，每个元素是一个record
    :param action: index指覆盖写入，update是添加写入
    :param id_key: 是指这个record的唯一id的名称，一般为_id,也可能为doc_id
    :return:
    """
    actions = []

    append_action = actions.append
    for d in tqdm(data, desc=f'{action} data'):
        if action == 'index':
            append_action({
                '_op_type': 'index',
                '_index': index_name,
                '_id': d[id_key],
                '_source': d
            })
        else:  # update
            append_action({
                "_op_type": "update",
                "doc_as_upsert": True,
                "_index": index_name,
                "_id": d[id_key],
                "doc": d,
            })
    bulk(es_client, actions, request_timeout=3600)
    es_client.indices.refresh(index=index_name)
    logger.success(f'{index_name} -- {action} -- {len(data)} records!')
    return


def es_iter_upload_data(es_client, index_name, data, action='update', id_key='_id'):
    # 单条数据的上传
    for d in tqdm(data, desc=f'{action} data'):
        if action == 'index':
            es_client.index(index=index_name, body=d, id=d[id_key], refresh=True)
        else:  # update
            es_client.update(index=index_name, body=d, id=d[id_key], refresh=True)

    logger.success(f'{index_name} -- {action} -- {len(data)} records!')

    return


def check_es_exist_index(es_client, index):
    # 是否存在index
    if not es_client.indices.exists(index=index):
        raise AssertionError(f'index: {index} not in es_client: {es_client}')


def get_alias(es_client, index):
    # index的别名
    return es_client.indices.get_alias(index=index)[index]


def es_count(es_client, index, query=None):
    # 计算index-query的数
    if not query:
        query = {
            "match_all": {}
        }
    return es_client.count(index=index, query=query)["count"]


def clear_index(es_client, index_name):
    """Clear all docs in the index."""
    es_client.delete_by_query(index_name, body={"query": {"match_all": {}}}, refresh=True)


def get_index_setting(es_client, index_name):
    """Return index settings."""
    return es_client.indices.get(index=index_name)


if __name__ == '__main__':
    pass

