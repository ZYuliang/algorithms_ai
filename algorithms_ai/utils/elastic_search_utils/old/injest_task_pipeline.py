import argparse
import os
import json
import sys
from elasticsearch7 import Elasticsearch
from common.logging_utils import get_logger
from es_injest_executor.config.es_config import settings

logger = get_logger(__name__)


def read_query_body(search_query_file):
    with open(search_query_file, "r", encoding="utf8") as f:
        return f.read()


def update_by_query_with_pipeline(es_index, search_body, pipeline_name):
    if settings.USE_SSL:
        logger.info("connect to es use ssl")
        es_client = Elasticsearch(
            hosts=[settings.ES_URL],
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=36000,
        )
    else:
        logger.info("connect to es without ssl")
        es_client = Elasticsearch(hosts=[settings.ES_URL], timeout=36000)

    logger.info(es_client.indices.get_alias(index=es_index))

    body = json.loads(search_body)

    injest_pipeline = es_client.ingest.get_pipeline(id=pipeline_name)

    if injest_pipeline.get(pipeline_name, ""):

        res = es_client.update_by_query(
            index=es_index, body=body, wait_for_completion=False, pipeline=pipeline_name, conflicts="proceed"
        )

        task_id = res.get("task", "")
        logger.info(f"Successfully submit Task: {task_id}, pipeline:{pipeline_name}, index:{es_index}")
        while True:
            try:
                task = es_client.tasks.get(
                    task_id=task_id, wait_for_completion=True, timeout="180s"
                )
                if task.get("completed", False):
                    logger.info(f"Task: {task_id} completed")
                    logger.info(json.dumps(task, indent=4))
                    return
            except Exception as e:
                logger.info("got exception for get task:")
                logger.info(e)
    else:
        raise ValueError(f"injest pipeline:{pipeline_name} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_body", type=str)
    parser.add_argument("--search_query_file", type=str)
    parser.add_argument("--es_index", type=str)
    parser.add_argument("--pipeline_name", type=str, required=True)

    args = parser.parse_args()

    if not args.search_body and not args.search_query_file:
        logger.warning(
            "empty search body and search query file, generating emtpy output"
        )
        os.makedirs(os.path.dirname(args.output_data_file), exist_ok=True)
        open(args.output_data_file, "w").close()
        sys.exit(0)

    if args.search_query_file and not args.search_body:
        logger.info(f"read search body from {args.search_query_file}")
        args.search_body = read_query_body(args.search_query_file)

    update_by_query_with_pipeline(
        es_index=args.es_index,
        search_body=args.search_body,
        pipeline_name=args.pipeline_name,
    )

