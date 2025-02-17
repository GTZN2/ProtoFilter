# -*- coding: utf-8 -*-

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough
from docx import Document
import pdfplumber
import csv
import json
import io
import numpy as np
import pandas as pd

import math

import openpyxl
import requests
import config

Baseurl = "https://chatapi.littlewheat.com"
Skey = config.Skey
url = Baseurl + "/v1/chat/completions"
headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {Skey}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
}

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""

        return text.strip()


def mahalanobis_distance(x, mu, sigma_inv):
    """
    计算点 x 到均值 mu 的马氏距离。

    参数:
    x (numpy.ndarray): 数据点，形状为 (n_features,)。
    mu (numpy.ndarray): 均值向量，形状为 (n_features,)。
    sigma_inv (numpy.ndarray): 协方差矩阵的逆，形状为 (n_features, n_features)。

    返回:
    float: 马氏距离。
    """
    delta = x - mu
    return np.sqrt(delta.T @ sigma_inv @ delta)

def LOC_compareDistance_for_wrongclass_to_add(list, llm_index):

    distance_right4 = math.sqrt((Proto_LOC_wrongclass_right[llm_index][0] - float(list[0])) ** 2 + (Proto_LOC_wrongclass_right[llm_index][1] - float(list[1])) ** 2)

    if distance_right4 > Proto_LOC_wrongclass_right_dis2pro[llm_index] and float(list[0]) >= Proto_LOC_wrongclass_right[llm_index][0] and float(list[1]) >= Proto_LOC_wrongclass_right[llm_index][1]:
        return True
    else:
        return False


def LOC_compareDistance_for_wrongclass_to_remove(list, llm_index,a):


    distance_right4 = math.sqrt((Proto_LOC_wrongclass_right[llm_index][0] - float(list[0])) ** 2 + (
                Proto_LOC_wrongclass_right[llm_index][1] - float(list[1])) ** 2)



    if distance_right4 > a*Proto_LOC_wrongclass_right_dis2pro[llm_index] and float(list[0]) < Proto_LOC_wrongclass_wrong[llm_index][0] and float(list[1]) < Proto_LOC_wrongclass_wrong[llm_index][1]:


        return True
    else:
        return False


def LOC_compareDistance_for_missMISC(list, llm_index,b):



    if Proto_LOC_missMISC_right[llm_index][0] - float(list[0])> b* Proto_LOC_missMISC_wrong_dis2pro[llm_index] :



        return True
    else:
        return False




def remove_subsets(strings):
    # 创建一个副本以避免在迭代时修改列表
    to_remove = []
    for i, s1 in enumerate(strings):
        for s2 in strings:
            if s1 != s2 and s1 in s2:
                to_remove.append(s1)
                break  # 无需继续检查，已经确定是子集

    # 使用集合去重，因为可能同一个元素被多次标记为子集
    to_remove = set(to_remove)

    # 创建新列表，排除需要删除的元素
    result = [s for s in strings if s not in to_remove]

    return result

a_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
b_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

#a_list = [0.1,0.2,0.3,0.4]

# 打开Excel文件
workbook = openpyxl.load_workbook('../dataset/conll03_test.xlsx')

# 选择工作表，假设是第一个工作表
sheet = workbook.active




# 预测了两次，一次LOC，一次MISC
Proto_LOC_wrongclass_right = [[9.5976431, 4.21212121]]
Proto_LOC_wrongclass_right_disp = [0.999997766612686]
Proto_LOC_wrongclass_right_dis2pro = [3.3210246156368295]
Proto_LOC_wrongclass_right_ni = [[[ 1.56975974, -0.04875409], [-0.04875409,  0.08240058]]]

Proto_LOC_wrongclass_wrong = [[8.9281768, 2.37845304]]
Proto_LOC_wrongclass_wrong_disp = [0.9998844618226922]
Proto_LOC_wrongclass_wrong_dis2pro = [2.3854715667852]
Proto_LOC_wrongclass_wrong_ni = [[[ 0.45191207, -0.03307334], [-0.03307334, 0.1491241 ]]]

Proto_LOC_missTarget_right = [[0, 3.3627451]]
Proto_LOC_missTarget_right_disp = [0.9999266781462995]

Proto_LOC_missTarget_wrong = [[0, 1.25640483]]
Proto_LOC_missTarget_wrong_disp = [0.6670173093263468]

Proto_LOC_missMISC_right = [[9.74130737, 0]]
Proto_LOC_missMISC_right_disp = [0.5881646389690773]
Proto_LOC_missMISC_right_dis2pro = [0.42167977855196015]

Proto_LOC_missMISC_wrong = [[9.2428884, 0 ]]
Proto_LOC_missMISC_wrong_disp = [0.7810876927774816]
Proto_LOC_missMISC_wrong_dis2pro = [0.9244382304918863]








# llm_list = ["gemma2:9b"]
#
# llm_onlyname_list = ["gemma2"]

# XXX_entity_num = 1578
# llm_XXX_entity_right_num =  1268
# llm_XXX_entity_num = 1608


# XXX_entity_num = 1632
# llm_XXX_entity_right_num =  1072
# llm_XXX_entity_num = 1655

for a in a_list:

    for b in b_list:
        # 创建一个 StringIO 对象
        output_buffer = io.StringIO()

        all_rule_list = []

        line_num = 1

        llm_LOC_entity_right_num = 0

        llm_LOC_entity_num = 0

        LOC_wrong_class_by_lmm = 0

        LOC_miss_by_lmm = 0

        LOC_inside_wrong_by_lmm = 0

        LOC_head_wrong_by_lmm = 0

        LOC_entity_num = 0

        wrong_rating_num = 0

        for row in sheet.iter_rows(min_row=2, values_only=True):

            print("line:" + str(line_num))

            increase_LOC_head_wrong_by_lmm1 = 0
            increase_LOC_head_wrong_by_lmm2 = 0

            # 计算总实体数
            #     if pd.notna(row[2]):
            #         cell_class = row[2].split(', ')
            #
            #         for i in cell_class:
            #             if i == 'B-LOC':
            #                 LOC_entity_num += 1
            #             elif i == 'B-LOC':
            #                 LOC_entity_num += 1
            #             elif i == 'B-LOC':
            #                 LOC_entity_num += 1
            #             elif i == 'B-MISC':
            #                 MISC_entity_num += 1
            #
            # print(LOC_entity_num)
            # print(LOC_entity_num)
            # print(LOC_entity_num)
            # print(MISC_entity_num)

            print("line:" + str(line_num), file=output_buffer)

            cell_sentence = row[0]  # 第一列的值
            if pd.isna(row[1]):
                cell_entity = []
            elif ', ' in row[1]:
                cell_entity = row[1].split(', ')
            else:
                cell_entity = str(row[1]).replace(".", "")

            if pd.isna(row[2]):
                cell_class = []
            elif ', ' in row[2]:
                cell_class = row[2].split(', ')
            else:
                cell_class = str(row[2])


            def remove_before_last_colon(s):
                index = s.rfind(':')
                if index != -1:
                    return s[index + 1:]
                return s


            def filter_non_int_convertible_elements(lst):
                # 收集不能转换为整数的元素的索引
                indices_to_remove = []
                for index, element in enumerate(lst):
                    try:
                        float(element)
                    except ValueError:
                        indices_to_remove.append(index)

                # 从后往前删除元素，以避免索引移动的问题
                for index in reversed(indices_to_remove):
                    del lst[index]

                return lst, indices_to_remove


            # belong to four types of entities: LOCson, location, location, and miscellaneous entity.

            # 创建提示模板
            template0 = """
                                        You are a helpful named entity recognition assistant.
                                        The following sentence may exist entities of the type "location".

                                        -If there are entities, please extract entities of the type "location" and only respond the extracted entities in the format: "Entity" for only one entity or "Entity, Entity" for two or more entities, without any other words.
                                        -If there are no corresponding entities, please respond with an empty string: "" without any other words.

                                        sentence：{sentence}
                                        """
            payload0 = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": template0.format(sentence=cell_sentence)
                    }
                ]
            })
            res0 = ""
            response0 = requests.request("POST", url, headers=headers, data=payload0)
            content_type0 = response0.headers.get("Content-Type", "")
            if "application/json" in content_type0:
                try:
                    res0 = response0.json()['choices'][0]['message']['content']
                except requests.exceptions.JSONDecodeError:
                    res0 = ""
            # response4 = requests.request("POST", url, headers=headers, data=payload4).json()['choices'][0]['message']['content']
            res0 = res0.replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*", "").replace(
                ".", "")
            res0 = remove_before_last_colon(res0)

            res0_list = []

            if 'This ' not in res0 and 'There ' not in res0 and res0 != '' and ' no ' not in res0:
                res0_list = [item.strip() for item in res0.split(',')]

                res0_list = [item for item in res0_list if item not in ('', []) and item is not None]

            print(res0_list)

            # 创建提示模板
            template1 = """
                                              Here is the entity class information: "location": This category includes names of specific geographical places, such as cities, countries, regions, or landmarks, within text.
                                              -Please rate the relevance of each phrase in the following list to the type "location" on a scale of 1 to 10.
                                              -Please only respond all entities in the list with rating strictly in the format: "Entity//Rating" for only one phrase or "Entity//Rating, Entity//Rating" for two or more phrases, without any other words.

                                              If the list is empty, please respond with an empty string: "" without any other words.

                                              list: {Entity_list}
                                            """

            template4 = """
                                                      Here is the entity class information: "location": This category includes names of specific geographical places, such as cities, countries, regions, or landmarks, within text.
                                                      The following sentence may contain entities other than those of the "location" type. 

                                                      If there are entities:
                                                      -Please extract all entities other than those of the "location" type.
                                                      -Please rate the relevance of extracted entities to the type "location" on a scale of 1 to 10.
                                                      -Please only respond all extracted entities with rating strictly in the format: "Entity//Rating" for only one phrase or "Entity//Rating, Entity//Rating" for two or more phrases, without any other words.

                                                      If there are no corresponding entities, please respond with an empty string: "" without any other words.

                                                      sentence：{sentence}
                                                      """

            # -If these noun phrases do not belong to the entity category in the question, please rate the lack of relevance between these noun phrases and the category in the question on a scale of -1 to -10.
            predict_LOC_entity_list = []
            predict_LOC_rating_list = []
            predict_MISC_entity_list = []
            predict_MISC_rating_list = []

            payload1 = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": template1.format(Entity_list=str(res0_list))
                    }
                ]
            })
            res1 = ""
            response1 = requests.request("POST", url, headers=headers, data=payload1)
            content_type1 = response1.headers.get("Content-Type", "")
            if "application/json" in content_type1:
                try:
                    res1 = response1.json()['choices'][0]['message']['content']
                except requests.exceptions.JSONDecodeError:
                    res1 = ""
            # response4 = requests.request("POST", url, headers=headers, data=payload4).json()['choices'][0]['message']['content']
            res1 = res1.replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*", "").replace(
                ".", "")

            print("LOC:", file=output_buffer)
            print(res1, file=output_buffer)

            print("LOC:")
            print(res1)

            res1 = remove_before_last_colon(res1)

            if 'This ' not in res1 and 'There ' not in res1 and res1 != '':
                res1_list = [item.strip() for item in res1.split(',')]

                res1_list = [item for item in res1_list if item not in ('', []) and item is not None]

                print(res1_list, file=output_buffer)
                print(res1_list)

                for predict_LOC in res1_list:
                    if "//" in predict_LOC:
                        if predict_LOC.split("//")[0] != "" and predict_LOC.split("//")[1] != "":
                            if predict_LOC.split("//")[0] not in predict_LOC_entity_list:
                                predict_LOC_entity_list.append(predict_LOC.split("//")[0])
                                predict_LOC_rating_list.append(predict_LOC.split("//")[1])

                predict_LOC_rating_list, LOC_indices_to_remove = filter_non_int_convertible_elements(
                    predict_LOC_rating_list)

                for index in reversed(LOC_indices_to_remove):
                    del predict_LOC_entity_list[index]

                for i in range(len(predict_LOC_rating_list)):
                    if float(predict_LOC_rating_list[i]) > 10:
                        predict_LOC_rating_list[i] = 10
                    elif float(predict_LOC_rating_list[i]) < 0:
                        predict_LOC_rating_list[i] = 0

            payload4 = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": template4.format(sentence=cell_sentence)
                    }
                ]
            })
            res4 = ""
            response4 = requests.request("POST", url, headers=headers, data=payload4)
            content_type4 = response4.headers.get("Content-Type", "")
            if "application/json" in content_type4:
                try:
                    res4 = response4.json()['choices'][0]['message']['content']
                except requests.exceptions.JSONDecodeError:
                    res4 = ""
            # response4 = requests.request("POST", url, headers=headers, data=payload4).json()['choices'][0]['message']['content']
            res4 = res4.replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*", "").replace(
                ".", "")

            print("MISC:", file=output_buffer)
            print(res4, file=output_buffer)

            print("MISC:")
            print(res4)

            res4 = remove_before_last_colon(res4)
            # llm_LOC_entity_results.append(res)

            if 'This ' not in res4 and 'There ' not in res4 and res4 != '':
                res4_list = [item.strip() for item in res4.split(',')]
                res4_list = [item for item in res4_list if item not in ('', []) and item is not None]

                print(res4_list, file=output_buffer)
                print(res4_list)

                for predict_MISC in res4_list:
                    if "//" in predict_MISC:
                        if predict_MISC.split("//")[0] != "" and predict_MISC.split("//")[1] != "":
                            if predict_MISC.split("//")[0] not in predict_MISC_entity_list:
                                predict_MISC_entity_list.append(predict_MISC.split("//")[0])
                                predict_MISC_rating_list.append(predict_MISC.split("//")[1])

                predict_MISC_rating_list, MISC_indices_to_remove = filter_non_int_convertible_elements(
                    predict_MISC_rating_list)

                for index in reversed(MISC_indices_to_remove):
                    del predict_MISC_entity_list[index]

                for i in range(len(predict_MISC_rating_list)):
                    if float(predict_MISC_rating_list[i]) > 10:
                        predict_MISC_rating_list[i] = 10
                    elif float(predict_MISC_rating_list[i]) < 0:
                        predict_MISC_rating_list[i] = 0

            final_LOC_prediction_list = predict_LOC_entity_list.copy()
            if len(predict_LOC_entity_list) > 0:
                for i in range(len(predict_LOC_entity_list)):

                    if predict_LOC_entity_list[i] in predict_MISC_entity_list:
                        MISC_index = predict_MISC_entity_list.index(predict_LOC_entity_list[i])

                        if LOC_compareDistance_for_wrongclass_to_remove(
                                [predict_LOC_rating_list[i], predict_MISC_rating_list[MISC_index]], 0, a):
                            final_LOC_prediction_list.remove(predict_LOC_entity_list[i])

                    elif LOC_compareDistance_for_missMISC([predict_LOC_rating_list[i], 0], 0, b):
                        final_LOC_prediction_list.remove(predict_LOC_entity_list[i])


            final_LOC_prediction_list = remove_subsets(final_LOC_prediction_list)

            if len(final_LOC_prediction_list) > 0:
                res1_list = final_LOC_prediction_list

                print(res1_list, file=output_buffer)
                print(res1_list)

                llm_LOC_entity_num += len(res1_list)
                checked_entity_in_prediction = []
                if "B-LOC" in cell_class:
                    indexes = []
                    cell_LOC_entity_head = []
                    for i in range(len(cell_class)):
                        if cell_class[i] == "B-LOC":
                            LOC_entity_num += 1
                            indexes.append(i)
                            cell_LOC_entity_head.append(cell_entity[i])
                            if cell_entity[i].lower() not in " ".join(res1_list).lower():
                                LOC_miss_by_lmm += 1


                    for r in res1_list:
                        if r.split()[0].lower() not in [s.lower() for s in cell_LOC_entity_head]:
                            LOC_wrong_class_by_lmm += 1

                    unchecked_index_prediction = res1_list.copy()

                    for j in range(len(indexes)):
                        for k in range(len(res1_list)):
                            is_wrong_class = False
                            if res1_list[k] not in checked_entity_in_prediction:
                                if cell_entity[indexes[j]].lower() in " ".join(res1_list).lower():
                                    if cell_entity[indexes[j]].lower() == res1_list[k].split()[0].lower():
                                        checked_entity_in_prediction.append(res1_list[k])

                                        is_wrong_class = True
                                        if len(res1_list[k].split()) > 1:
                                            if len(cell_entity) > indexes[j] + 1:
                                                if cell_class[indexes[j] + 1] == 'I-LOC':

                                                    if cell_entity[indexes[j] + 1].lower() == res1_list[k].split()[
                                                        1].lower():
                                                        llm_LOC_entity_right_num += 1
                                                        is_wrong_class = False

                                                    else:
                                                        LOC_inside_wrong_by_lmm += 1
                                                        is_wrong_class = False

                                                else:
                                                    LOC_inside_wrong_by_lmm += 1
                                                    is_wrong_class = False

                                            else:
                                                LOC_inside_wrong_by_lmm += 1
                                                is_wrong_class = False

                                        elif len(cell_entity) > indexes[j] + 1:
                                            if cell_class[indexes[j] + 1] == 'I-LOC':
                                                LOC_inside_wrong_by_lmm += 1
                                                is_wrong_class = False

                                            else:
                                                llm_LOC_entity_right_num += 1
                                                is_wrong_class = False

                                        else:
                                            llm_LOC_entity_right_num += 1
                                            is_wrong_class = False

                                    elif len(res1_list[k].split()) > 1:
                                        if cell_entity[indexes[j]].lower() == res1_list[k].split()[1].lower():
                                            if res1_list[k].split()[0].lower() not in " ".join(res1_list).lower():
                                                increase_LOC_head_wrong_by_lmm2 += 1
                                                checked_entity_in_prediction.append(res1_list[k])


                            if is_wrong_class:
                                LOC_wrong_class_by_lmm += 1



                elif len(res1_list) > 0:
                    LOC_wrong_class_by_lmm += len(res1_list)

                if "I-LOC" in cell_class:
                    indexes_inside = []
                    for i in range(len(cell_class)):
                        if cell_class[i] == "I-LOC":
                            indexes_inside.append(i)

                    for inside_entity_index in range(len(indexes_inside)):
                        if cell_entity[indexes_inside[inside_entity_index]].lower() in " ".join(res1_list).lower():
                            for j in res1_list:
                                if j not in checked_entity_in_prediction:
                                    if cell_entity[indexes_inside[inside_entity_index]].lower() == j.split()[0].lower():
                                        increase_LOC_head_wrong_by_lmm1 += 1
                                        checked_entity_in_prediction.append(j)
                                        if cell_entity[indexes_inside[inside_entity_index] - 1].lower() not in [
                                            s.split()[0].lower() for s in res1_list]:
                                            LOC_miss_by_lmm -= 1
                                        break

                LOC_head_wrong_by_lmm += increase_LOC_head_wrong_by_lmm1 + increase_LOC_head_wrong_by_lmm2

            elif "B-LOC" in cell_class:
                for i in range(len(cell_class)):
                    if cell_class[i] == "B-LOC":
                        LOC_miss_by_lmm += 1
                        LOC_entity_num += 1
            # 重复计算
            LOC_wrong_class_by_lmm -= increase_LOC_head_wrong_by_lmm1

            print("LOC_entity_num: " + str(LOC_entity_num))
            print("llm_LOC_entity_right_num: " + str(llm_LOC_entity_right_num))
            print("llm_LOC_entity_num: " + str(llm_LOC_entity_num))
            print("LOC_wrong_class_by_lmm: " + str(LOC_wrong_class_by_lmm))
            print("LOC_inside_wrong_by_lmm: " + str(LOC_inside_wrong_by_lmm))
            print("LOC_head_wrong_by_lmm: " + str(LOC_head_wrong_by_lmm))
            print("LOC_miss_by_lmm: " + str(LOC_miss_by_lmm))
            print("wrong_rating_num: " + str(wrong_rating_num))

            print("---" * 30)

            print("LOC_entity_num: " + str(LOC_entity_num), file=output_buffer)
            print("llm_LOC_entity_right_num: " + str(llm_LOC_entity_right_num), file=output_buffer)
            print("llm_LOC_entity_num: " + str(llm_LOC_entity_num), file=output_buffer)
            print("LOC_wrong_class_by_lmm: " + str(LOC_wrong_class_by_lmm), file=output_buffer)
            print("LOC_inside_wrong_by_lmm: " + str(LOC_inside_wrong_by_lmm), file=output_buffer)
            print("LOC_head_wrong_by_lmm: " + str(LOC_head_wrong_by_lmm), file=output_buffer)
            print("LOC_miss_by_lmm: " + str(LOC_miss_by_lmm), file=output_buffer)
            print("---" * 30, file=output_buffer)

            line_num += 1

        P = llm_LOC_entity_right_num / llm_LOC_entity_num

        R = llm_LOC_entity_right_num / LOC_entity_num

        f1 = (2 * P * R) / (P + R)

        print("f1: " + str(f1), file=output_buffer)
        print("f1: " + str(f1))
        # 获取 StringIO 对象的内容
        output_str = output_buffer.getvalue()

        # 将内容写入文件
        with open(r"C:\NER\results\LOC_result_with_rule_entitywise_" + str(
                a) + "\CONLL03_validset_gpt3.5_LOC_withfilter_result_with_describe_0_shot" + str(a) + ".txt", "w",
                  encoding="utf-8") as file:
            file.write(output_str)

        # 关闭 StringIO 对象
        output_buffer.close()














