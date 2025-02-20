# -*- coding: utf-8 -*-


from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
import numpy as np
import pandas as pd

import math

import openpyxl


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""

        return text.strip()


def mahalanobis_distance(x, mu, sigma_inv):

    delta = x - mu
    return np.sqrt(delta.T @ sigma_inv @ delta)

def ORG_compareDistance_for_wrongclass_to_add(list, llm_index):

    distance_right4 = math.sqrt((Proto_ORG_wrongclass_right[llm_index][0] - float(list[0])) ** 2 + (Proto_ORG_wrongclass_right[llm_index][1] - float(list[1])) ** 2)

    if distance_right4 > Proto_ORG_wrongclass_right_dis2pro[llm_index] and float(list[0]) >= Proto_ORG_wrongclass_right[llm_index][0] and float(list[1]) >= Proto_ORG_wrongclass_right[llm_index][1]:
        return True
    else:
        return False


def ORG_compareDistance_for_wrongclass_to_remove(list, llm_index,a):


    distance_right4 = math.sqrt((Proto_ORG_wrongclass_right[llm_index][0] - float(list[0])) ** 2 + (
                Proto_ORG_wrongclass_right[llm_index][1] - float(list[1])) ** 2)

    if distance_right4 > a*Proto_ORG_wrongclass_right_dis2pro[llm_index] and float(list[0]) < Proto_ORG_wrongclass_wrong[llm_index][0] and float(list[1]) < Proto_ORG_wrongclass_wrong[llm_index][1]:


        return True
    else:
        return False


def ORG_compareDistance_for_missMISC(list, llm_index,b):

    if Proto_ORG_missMISC_right[llm_index][0] - float(list[0])> b* Proto_ORG_missMISC_wrong_dis2pro[llm_index] :

        return True
    else:
        return False



def remove_subsets(strings):

    to_remove = []
    for i, s1 in enumerate(strings):
        for s2 in strings:
            if s1 != s2 and s1 in s2:
                to_remove.append(s1)
                break

    to_remove = set(to_remove)


    result = [s for s in strings if s not in to_remove]

    return result


def remove_before_last_colon(s):
    index = s.rfind(':')
    if index != -1:
        return s[index + 1:]
    return s


def filter_non_int_convertible_elements(lst):
    indices_to_remove = []
    for index, element in enumerate(lst):
        try:
            float(element)
        except ValueError:
            indices_to_remove.append(index)

    for index in reversed(indices_to_remove):
        del lst[index]

    return lst, indices_to_remove

a_list = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
b_list = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

workbook = openpyxl.load_workbook('../dataset/conll03_test.xlsx')


sheet = workbook.active

llm_list = ["gemma2:9b"]

llm_onlyname_list = ["gemma2"]


# Prototypes, distance
Proto_ORG_wrongclass_right = [[9.39290407, 8.19973719]]
Proto_ORG_wrongclass_right_disp = [0.9996806478941954]
Proto_ORG_wrongclass_right_dis2pro = [2.233403786036866]
Proto_ORG_wrongclass_right_ni = [[[ 1.01564769, -0.12269955], [-0.12269955 , 0.158446  ]]]

Proto_ORG_wrongclass_wrong = [[8.50253807, 7.47461929]]
Proto_ORG_wrongclass_wrong_disp = [0.9999198810232676]
Proto_ORG_wrongclass_wrong_dis2pro = [2.4976063580085026]
Proto_ORG_wrongclass_wrong_ni = [[[ 0.40300036, -0.16773036],[-0.16773036 , 0.23860663]]]

Proto_ORG_missTarget_right = [[0,  6.88421053]]
Proto_ORG_missTarget_right_disp = [0.999984197157546]

Proto_ORG_missTarget_wrong = [[0,   5.20168776]]
Proto_ORG_missTarget_wrong_disp = [0.9999076064738645]

Proto_ORG_missMISC_right = [[9.54446855, 0]]
Proto_ORG_missMISC_right_disp = [0.7155460889425672]
Proto_ORG_missMISC_right_dis2pro = [0.6561233948645078]

Proto_ORG_missMISC_wrong = [[8.55772994, 0]]
Proto_ORG_missMISC_wrong_disp = [0.9812375092701624]
Proto_ORG_missMISC_wrong_dis2pro = [1.4610391351136065]






for a in a_list:
    for b in b_list:
        for llm_index in range(len(llm_list)):
            llm = ChatOllama(model=llm_list[llm_index])

            all_rule_list = []

            line_num = 1

            llm_ORG_entity_right_num = 0

            llm_ORG_entity_num = 0

            ORG_wrong_class_by_lmm = 0

            ORG_miss_by_lmm = 0

            ORG_inside_wrong_by_lmm = 0

            ORG_head_wrong_by_lmm = 0

            ORG_entity_num = 0

            wrong_rating_num = 0

            for row in sheet.iter_rows(min_row=2, values_only=True):

                print("line:" + str(line_num))

                increase_ORG_head_wrong_by_lmm1 = 0
                increase_ORG_head_wrong_by_lmm2 = 0

                cell_sentence = row[0]
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


                template0 = """
                           You are a helpful named entity recognition assistant.
                           The following sentence may exist entities of the type "organization".

                           -If there are entities, please extract entities of the type "organization" and only respond the extracted entities in the format: "Entity" for only one entity or "Entity, Entity" for two or more entities, without any other words.
                           -If there are no corresponding entities, please respond with an empty string: "" without any other words.

                           sentence：{sentence}
                           """
                prompt0 = ChatPromptTemplate.from_template(template0)
                output_parser = CommaSeparatedListOutputParser()
                chain = prompt0 | llm | output_parser
                res0 = chain.invoke({"sentence": cell_sentence}).replace('"', '').replace("'", "").replace("\n","").replace("* ", "").replace("*", "").replace(".", "")
                res0 = remove_before_last_colon(res0)

                res0_list = []

                if 'This ' not in res0 and 'There ' not in res0 and res0 != '' and ' no ' not in res0:
                    res0_list = [item.strip() for item in res0.split(',')]

                    res0_list = [item for item in res0_list if item not in ('', []) and item is not None]

                print(res0_list)


                template1 = """
                              Here is the entity class information: "organization": This category includes names of formally structured groups, such as companies, institutions, agencies, or teams, within text.
                              -Please rate the relevance of each phrase in the following list to the type "organization" on a scale of 1 to 10.
                              -Please only respond all entities in the list with rating strictly in the format: "Entity//Rating" for only one phrase or "Entity//Rating, Entity//Rating" for two or more phrases, without any other words.

                              If the list is empty, please respond with an empty string: "" without any other words.

                              list: {Entity_list}

                              """

                template4 = """
                              Here is the entity class information: "organization": This category includes names of formally structured groups, such as companies, institutions, agencies, or teams, within text.
                              The following sentence may contain entities other than those of the "organization" type. 

                              If there are entities:
                              -Please extract all entities other than those of the "organization" type.
                              -Please rate the relevance of extracted entities to the type "organization" on a scale of 1 to 10.
                              -Please only respond all extracted entities with rating strictly in the format: "Entity//Rating" for only one phrase or "Entity//Rating, Entity//Rating" for two or more phrases, without any other words.

                              If there are no corresponding entities, please respond with an empty string: "" without any other words.

                              sentence：{sentence}

                              """

                predict_ORG_entity_list = []
                predict_ORG_rating_list = []
                predict_MISC_entity_list = []
                predict_MISC_rating_list = []

                prompt1 = ChatPromptTemplate.from_template(template1)
                output_parser = CommaSeparatedListOutputParser()
                chain1 = prompt1 | llm | output_parser
                res1 = chain1.invoke({"Entity_list": str(res0_list)}).replace('"', '').replace("'", "").replace("\n",
                                                                                                                "").replace(
                    "* ", "").replace("*", "").replace(".", "")


                print("ORG:")
                print(res1)

                res1 = remove_before_last_colon(res1)

                if 'This ' not in res1 and 'There ' not in res1 and res1 != '':
                    res1_list = [item.strip() for item in res1.split(',')]

                    res1_list = [item for item in res1_list if item not in ('', []) and item is not None]

                    print(res1_list)

                    for predict_ORG in res1_list:
                        if "//" in predict_ORG:
                            if predict_ORG.split("//")[0] != "" and predict_ORG.split("//")[1] != "":
                                if predict_ORG.split("//")[0] not in predict_ORG_entity_list:
                                    predict_ORG_entity_list.append(predict_ORG.split("//")[0])
                                    predict_ORG_rating_list.append(predict_ORG.split("//")[1])

                    predict_ORG_rating_list, ORG_indices_to_remove = filter_non_int_convertible_elements(
                        predict_ORG_rating_list)

                    for index in reversed(ORG_indices_to_remove):
                        del predict_ORG_entity_list[index]

                    for i in range(len(predict_ORG_rating_list)):
                        if float(predict_ORG_rating_list[i]) > 10:
                            predict_ORG_rating_list[i] = 10
                        elif float(predict_ORG_rating_list[i]) < 0:
                            predict_ORG_rating_list[i] = 0

                prompt4 = ChatPromptTemplate.from_template(template4)
                output_parser = CommaSeparatedListOutputParser()
                chain4 = prompt4 | llm | output_parser
                res4 = chain4.invoke({"sentence": cell_sentence}).replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*", "").replace(".", "")

                print("MISC:")
                print(res4)

                res4 = remove_before_last_colon(res4)

                if 'This ' not in res4 and 'There ' not in res4 and res4 != '':
                    res4_list = [item.strip() for item in res4.split(',')]
                    res4_list = [item for item in res4_list if item not in ('', []) and item is not None]

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

                final_ORG_prediction_list = predict_ORG_entity_list.copy()
                if len(predict_ORG_entity_list) > 0:
                    for i in range(len(predict_ORG_entity_list)):

                        if predict_ORG_entity_list[i] in predict_MISC_entity_list:
                            MISC_index = predict_MISC_entity_list.index(predict_ORG_entity_list[i])

                            if ORG_compareDistance_for_wrongclass_to_remove(
                                    [predict_ORG_rating_list[i], predict_MISC_rating_list[MISC_index]], llm_index, a):
                                final_ORG_prediction_list.remove(predict_ORG_entity_list[i])

                        elif ORG_compareDistance_for_missMISC([predict_ORG_rating_list[i], 0], llm_index, b):
                            final_ORG_prediction_list.remove(predict_ORG_entity_list[i])


                final_ORG_prediction_list = remove_subsets(final_ORG_prediction_list)

                if len(final_ORG_prediction_list) > 0:
                    res1_list = final_ORG_prediction_list

                    print(res1_list)

                    llm_ORG_entity_num += len(res1_list)
                    checked_entity_in_prediction = []
                    if "B-ORG" in cell_class:
                        indexes = []
                        cell_ORG_entity_head = []
                        for i in range(len(cell_class)):
                            if cell_class[i] == "B-ORG":
                                ORG_entity_num += 1
                                indexes.append(i)
                                cell_ORG_entity_head.append(cell_entity[i])
                                if cell_entity[i].lower() not in " ".join(res1_list).lower():
                                    ORG_miss_by_lmm += 1

                        for r in res1_list:
                            if r.split()[0].lower() not in [s.lower() for s in cell_ORG_entity_head]:
                                ORG_wrong_class_by_lmm += 1

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

                                                    if cell_class[indexes[j] + 1] == 'I-ORG':

                                                        if cell_entity[indexes[j] + 1].lower() == res1_list[k].split()[
                                                            1].lower():
                                                            llm_ORG_entity_right_num += 1
                                                            is_wrong_class = False

                                                        else:
                                                            ORG_inside_wrong_by_lmm += 1
                                                            is_wrong_class = False

                                                    else:
                                                        ORG_inside_wrong_by_lmm += 1
                                                        is_wrong_class = False

                                                else:
                                                    ORG_inside_wrong_by_lmm += 1
                                                    is_wrong_class = False

                                            elif len(cell_entity) > indexes[j] + 1:
                                                if cell_class[indexes[j] + 1] == 'I-ORG':
                                                    ORG_inside_wrong_by_lmm += 1
                                                    is_wrong_class = False

                                                else:
                                                    llm_ORG_entity_right_num += 1
                                                    is_wrong_class = False

                                            else:
                                                llm_ORG_entity_right_num += 1
                                                is_wrong_class = False

                                        elif len(res1_list[k].split()) > 1:
                                            if cell_entity[indexes[j]].lower() == res1_list[k].split()[1].lower():
                                                if res1_list[k].split()[0].lower() not in " ".join(res1_list).lower():
                                                    increase_ORG_head_wrong_by_lmm2 += 1
                                                    checked_entity_in_prediction.append(res1_list[k])


                                if is_wrong_class:
                                    ORG_wrong_class_by_lmm += 1



                    elif len(res1_list) > 0:
                        ORG_wrong_class_by_lmm += len(res1_list)

                    if "I-ORG" in cell_class:
                        indexes_inside = []
                        for i in range(len(cell_class)):
                            if cell_class[i] == "I-ORG":
                                indexes_inside.append(i)

                        for inside_entity_index in range(len(indexes_inside)):
                            if cell_entity[indexes_inside[inside_entity_index]].lower() in " ".join(res1_list).lower():
                                for j in res1_list:
                                    if j not in checked_entity_in_prediction:
                                        if cell_entity[indexes_inside[inside_entity_index]].lower() == j.split()[
                                            0].lower():
                                            increase_ORG_head_wrong_by_lmm1 += 1
                                            checked_entity_in_prediction.append(j)
                                            if cell_entity[indexes_inside[inside_entity_index] - 1].lower() not in [
                                                s.split()[0].lower() for s in res1_list]:
                                                ORG_miss_by_lmm -= 1
                                            break

                    ORG_head_wrong_by_lmm += increase_ORG_head_wrong_by_lmm1 + increase_ORG_head_wrong_by_lmm2

                elif "B-ORG" in cell_class:
                    for i in range(len(cell_class)):
                        if cell_class[i] == "B-ORG":
                            ORG_miss_by_lmm += 1
                            ORG_entity_num += 1

                ORG_wrong_class_by_lmm -= increase_ORG_head_wrong_by_lmm1

                print("ORG_entity_num: " + str(ORG_entity_num))
                print("llm_ORG_entity_right_num: " + str(llm_ORG_entity_right_num))
                print("llm_ORG_entity_num: " + str(llm_ORG_entity_num))
                print("ORG_wrong_class_by_lmm: " + str(ORG_wrong_class_by_lmm))
                print("ORG_inside_wrong_by_lmm: " + str(ORG_inside_wrong_by_lmm))
                print("ORG_head_wrong_by_lmm: " + str(ORG_head_wrong_by_lmm))
                print("ORG_miss_by_lmm: " + str(ORG_miss_by_lmm))
                print("wrong_rating_num: " + str(wrong_rating_num))
                print("---" * 30)

                line_num += 1

            P = llm_ORG_entity_right_num / llm_ORG_entity_num

            R = llm_ORG_entity_right_num / ORG_entity_num

            f1 = (2 * P * R) / (P + R)

            print("f1: " + str(f1))


























