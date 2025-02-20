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


def LOC_compareDistance_for_wrongclass_to_add(list, llm_index):
    distance_right4 = math.sqrt((Proto_LOC_wrongclass_right[llm_index][0] - float(list[0])) ** 2 + (
                Proto_LOC_wrongclass_right[llm_index][1] - float(list[1])) ** 2)

    if distance_right4 > Proto_LOC_wrongclass_right_dis2pro[llm_index] and float(list[0]) >= \
            Proto_LOC_wrongclass_right[llm_index][0] and float(list[1]) >= Proto_LOC_wrongclass_right[llm_index][1]:
        return True
    else:
        return False

def LOC_compareDistance_for_wrongclass_to_remove(list, llm_index, a):


    distance_right4 = math.sqrt((Proto_LOC_wrongclass_right[llm_index][0] - float(list[0])) ** 2 + (
            Proto_LOC_wrongclass_right[llm_index][1] - float(list[1])) ** 2)

    if distance_right4 > a * Proto_LOC_wrongclass_right_dis2pro[llm_index] and float(list[0]) < \
            Proto_LOC_wrongclass_wrong[llm_index][0] and float(list[1]) < Proto_LOC_wrongclass_wrong[llm_index][1]:


        return True
    else:
        return False

def LOC_compareDistance_for_missMISC(list, llm_index, b):

    if Proto_LOC_missMISC_right[llm_index][0] - float(list[0]) > b * Proto_LOC_missMISC_wrong_dis2pro[llm_index]:

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


# hyperparameters
a_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
b_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# dataset
workbook = openpyxl.load_workbook('../dataset/conll03_test.xlsx')
sheet = workbook.active

# LLM list
llm_list = ["gemma2:9b"]
llm_onlyname_list = ["gemma2"]

# prototypes
Proto_LOC_wrongclass_right = [[9.63063063, 8.03153153]]
Proto_LOC_wrongclass_right_disp = [0.9985650769538227]
Proto_LOC_wrongclass_right_dis2pro = [2.066736942729236]
Proto_LOC_wrongclass_right_ni = [[[ 2.16210136, -0.18236831], [-0.18236831,  0.1809528 ]]]

Proto_LOC_wrongclass_wrong = [[9.28595601, 7.41624365]]
Proto_LOC_wrongclass_wrong_disp = [0.9999909922792936]
Proto_LOC_wrongclass_wrong_dis2pro = [ 2.8802625419095143]
Proto_LOC_wrongclass_wrong_ni = [[[ 0.66162543, -0.06722696], [-0.06722696,  0.10680367]]]

Proto_LOC_missTarget_right = [[0, 7.45]]
Proto_LOC_missTarget_right_disp = [0.99901365035523]

Proto_LOC_missTarget_wrong = [[0, 3.99102285]]
Proto_LOC_missTarget_wrong_disp = [0.9999365346753558]

Proto_LOC_missMISC_right = [[9.84417965, 0]]
Proto_LOC_missMISC_right_disp = [0.5673655323284756]
Proto_LOC_missMISC_right_dis2pro = [0.27536354860743]

Proto_LOC_missMISC_wrong = [[9.26851852, 0]]
Proto_LOC_missMISC_wrong_disp = [0.8853083837644604]
Proto_LOC_missMISC_wrong_dis2pro = [0.9295910493827159]



for a in a_list:
    for b in b_list:
        for llm_index in range(len(llm_list)):
            llm = ChatOllama(model=llm_list[llm_index])

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
                        The following sentence may exist entities of the type "location".

                        -If there are entities, please extract entities of the type "location" and only respond the extracted entities in the format: "Entity" for only one entity or "Entity, Entity" for two or more entities, without any other words.
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

                predict_LOC_entity_list = []
                predict_LOC_rating_list = []
                predict_MISC_entity_list = []
                predict_MISC_rating_list = []

                prompt1 = ChatPromptTemplate.from_template(template1)
                output_parser = CommaSeparatedListOutputParser()
                chain1 = prompt1 | llm | output_parser
                res1 = chain1.invoke({"Entity_list": str(res0_list)}).replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*", "").replace(".", "")


                print("LOC:")
                print(res1)

                res1 = remove_before_last_colon(res1)

                if 'This ' not in res1 and 'There ' not in res1 and res1 != '':
                    res1_list = [item.strip() for item in res1.split(',')]

                    res1_list = [item for item in res1_list if item not in ('', []) and item is not None]

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

                prompt4 = ChatPromptTemplate.from_template(template4)
                output_parser = CommaSeparatedListOutputParser()
                chain4 = prompt4 | llm | output_parser
                res4 = chain4.invoke({"sentence": cell_sentence}).replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*", "").replace(".", "")

                print("MISC:")
                print(res4)

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

                final_LOC_prediction_list = predict_LOC_entity_list.copy()
                if len(predict_LOC_entity_list) > 0:
                    for i in range(len(predict_LOC_entity_list)):

                        if predict_LOC_entity_list[i] in predict_MISC_entity_list:
                            MISC_index = predict_MISC_entity_list.index(predict_LOC_entity_list[i])

                            if LOC_compareDistance_for_wrongclass_to_remove(
                                    [predict_LOC_rating_list[i], predict_MISC_rating_list[MISC_index]], llm_index, a):
                                final_LOC_prediction_list.remove(predict_LOC_entity_list[i])

                        elif LOC_compareDistance_for_missMISC([predict_LOC_rating_list[i], 0], llm_index, b):
                            final_LOC_prediction_list.remove(predict_LOC_entity_list[i])


                final_LOC_prediction_list = remove_subsets(final_LOC_prediction_list)

                if len(final_LOC_prediction_list) > 0:
                    res1_list = final_LOC_prediction_list

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
                                        if cell_entity[indexes_inside[inside_entity_index]].lower() == j.split()[
                                            0].lower():
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

                line_num += 1

            P = llm_LOC_entity_right_num / llm_LOC_entity_num

            R = llm_LOC_entity_right_num / LOC_entity_num

            f1 = (2 * P * R) / (P + R)

            print("f1: " + str(f1))
























