# 创建
需要mapping ，需要properties，常用`{"type": "text","fields": {"keyword": {"type": "keyword","ignore_above": 256}}`表示既支持text-match模糊搜索，也支持keyword-term搜索，term-keyword搜索比text块
```
PUT /cplatform_article_pubmed_analysis
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "final_pipeline": "timestamp",
    "refresh_interval": "30s"
  },
  "mappings": {
      "properties": {
        "_timestamp": {
          "type": "date"
        },
        "_update_by_query_no": {
          "type": "long"
        },
        "abstract": {
          "type": "nested",
          "include_in_root": true,
          "properties": {
            "Label": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 256
                }
              }
            },
            "NlmCategory": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 256
                }
              }
            }

```


# 迁移数据从一个索引到另一个索引
从 ’cplatform_article_pubmed_3‘ 到cplatform_article_pubmed_analysis。注意添加搜索条件进行部分迁移
```
POST _reindex?slices=auto&wait_for_completion=false&requests_per_second=1000
{
  "conflicts": "proceed", 
  "source": {
    "index": "cplatform_article_pubmed_3",
    "query": {
      "bool": {
        "must": [
          {
            "bool": {
              "must": [
                {
                  "range": {
                    "pub_date.pubmed.year": {
                      "gte": 1990,
                      "lte": 2023,
                      "boost": 2
                    }
                  }
                },
                {
                  "terms": {
                    "journal_impact_factor.cas_major_quartile": [
                      1
                    ]
                  }
                },
                {
                  "bool": {
                    "must_not": {
                      "term": {
                        "deleted": true
                      }
                    }
                  }
                }
              ]
            }
          }
        ]
      }
    },
    "_source": {
      "includes": [
        "_id",
        "title",
        "article_journal",
        "journal_impact_factor",
        "pub_date.pubmed.date",
        "citedby_calculated.count",
        "abstract",
        "keywords",
        "mesh_terms"
        ],
      "excludes": []
    }
  },
  "dest": {
    "index": "cplatform_article_pubmed_analysis"
  }
}
```


# 搜索
_search 表示搜索，from偏移量，从第几个开始，一般为0，size最大的返回数据的大小，默认为10,track_total_hits返回的最大映射数，不设置为True则最大返回10000，设置为True才返回所有的count,source表示必须返回的字段，source-include表示必须返回的字段，source-excludes表示不要返回的字段，fields表示要统计的字段--这个的返回值在fields中，sort表示排序的字段可以实时计算，search-after大数据返回

must必须，must_not必须不，should或者，term某个keyword，terms可以输入多个keyword，range范围,exists字段存在（不为null和[]）

`"query": {
    "match_all": {}
  }` 表示搜索全部


```
GET /cplatform_article_pubmed_3/_search/
{
  "query": {
    "bool": {
      "must": [
        {
          "bool": {
            "must": [
              {
                "range": {
                  "pub_date.pubmed.year": {
                    "gte": 1990,
                    "lte": 2023,
                    "boost": 2
                  }
                }
              },
               {
          "terms": {
            "entity.entity_type.keyword": ["target"]
          }
        },
              {
                "bool": {
                  "must_not": {
                    "exists": {
                      "field": "entity"
                    }
                  }
                }
              },
              {
                "terms": {
                  "journal_impact_factor.cas_major_quartile": [
                    1
                  ]
                }
              },
              {
                "bool": {
                  "must_not": {
                    "term": {
                      "deleted": true
                    }
                  }
                }
              },
              {
                "terms": {
                  "keywords.keyword": [
                    "Mono-ARTDs",
                    "human"
                  ]
                }
              }
              
            ]
          }
        }
      ]
    }
  },
  "_source": {
    "includes": [
      "_id",
      "keywords"
      ],
    "excludes": []
  },
  "fields": [
    "citedby_calculated.count",
    "monthly_citation_count"
  ],
  "from": 0,
  "track_total_hits": true,
   
 "sort": {
  "_script": {
    "type": "number",
    "script": {
      "lang": "painless",
      "source": "doc['citedby_calculated.count'].value * params.factor + doc['monthly_citation_count'][0] * params.factor_2 + doc['journal_impact_factor.value'].value * params.factor_3",
      "params": {
        "factor": 1.1,
        "factor_2":2,
        "factor_3":2
      }
    },
    "order": "desc"
  }
}
```

查询所有文档数
```
GET cplatform_article_pubmed_analysis/_search
{
  "track_total_hits": true,
  "size": 0
}
```

计数
```
GET cplatform_article_pubmed_analysis/_count
```

# 添加
在mapping中添加字段
```
PUT /cplatform_article_pubmed_analysis/_mapping
{
  "properties": {
    "entity":{
      "type": "nested",
      "include_in_root": true,
      "properties": {
        "entity_type":{
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword",
              "ignore_above": 256
            }
          }
        },
        "section": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword",
              "ignore_above": 256
            }
          }
        },
        "start_offset": {
          "type": "long"
        },
        "end_offset": {
          "type": "long"
        },
        "text": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword",
              "ignore_above": 256
            }
          }
        },
        "linking":{
          "type":"nested",
          "include_in_root": true,
          "properties":{
            "dict_id": {
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 256
                }
              }
            },
            "entry_name":{
              "type": "text",
              "fields": {
                "keyword": {
                  "type": "keyword",
                  "ignore_above": 256
                }
              }
            }
          }
        }
      }
    }
  } 
}
```

添加一个实时计算的字段
```
PUT /cplatform_article_pubmed_analysis/_mapping
{
 "runtime": {
      "monthly_citation_count": {
        "type": "double",
        "script": {
          "source": "\n          if (!doc['citedby_calculated.count'].empty && !doc['pub_date.pubmed.date'].empty) {\n            long diff = new Date().getTime() / 1000 - doc['pub_date.pubmed.date'].value.toEpochSecond();\n            //double days = diff / 86400.0;\n            //emit(doc['citedby_calculated.count'].value / days);\n            Calendar cal = Calendar.getInstance();\n            cal.setTimeInMillis(diff * 1000);\n            double months = Math.max(1, (cal.get(Calendar.YEAR)-1970)*12+cal.get(Calendar.MONTH));\n            emit(doc['citedby_calculated.count'].value / months);\n          }\n          ",
          "lang": "painless"
        }
      }
    }

}

```



# 任务
查询任务
```
GET _tasks/klxx3ydtQtiXIuaIFGO9fQ:588249939
```
取消任务
```
POST _tasks/klxx3ydtQtiXIuaIFGO9fQ:588249939/_cancel
```





# 聚合
```
{
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "pub_date.pubmed.date": {
                                "gte": time_start,
                                "lte": time_end
                            }
                        }
                    },
                    {
                        "bool": {
                            "should": [
                            ]
                        }
                    }
                ]
            }
        },
        "size": 0,
        "aggs": {
            "agg_keywords": {
                "date_histogram": {
                    "field": "pub_date.pubmed.date",
                    "calendar_interval": time_interval,
                    "format": "yyyy-MM",
                    "min_doc_count": 0
                }
            }
        }
    }`

```



# 删除
删除索引
```
DELETE /cplatform_article_pubmed_analysis
```

删除字段
```
POST /cplatform_article_pubmed_analysis/_update_by_query
{
  "script": {
    "source": "ctx._source.remove(\"disease_entry\")"
  },
  "query" : {
      "exists": { "field": "disease_entry" }
  }
}
```





