# -*- coding: utf-8 -*-

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
import pandas as pd
import openpyxl


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""

        return text.strip()

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


def RatingCollect(llm_list,entity_type,sheet,label_descri):

    label = entity_type[:3].upper()

    for llm_index in range(len(llm_list)):
        llm = ChatOllama(model=llm_list[llm_index])

        all_rule_list = []

        line_num = 1

        for row in sheet.iter_rows(min_row=2, values_only=True):

            print("line:" + str(line_num))

            cell_sentence = row[0]
            if pd.isna(row[1]):
                cell_entity = []
            elif ', ' in row[1]:
                cell_entity = row[1].split(', ')
            else:
                cell_entity = str(row[1])

            if pd.isna(row[2]):
                cell_class = []
            elif ', ' in row[2]:
                cell_class = row[2].split(', ')
            else:
                cell_class = str(row[2])


            template1 = """
                          The following sentence may exist entities of the type {type}. Here is the entity type information: {type}: {label_descri}

                          If there are entities:
                          -Please extract all entities of the type {type}.
                          -Please rate the relevance of extracted entities to the type {type} on a scale of 1 to 10.
                          -Please only respond all extracted entities with rating strictly in the format: "Entity//Rating" for only one phrase or "Entity//Rating, Entity//Rating" for two or more phrases, without any other words.

                          If there are no corresponding entities, please respond with an empty string: "" without any other words.

                          sentence：{sentence}

                          """

            template4 = """
                          The following sentence may contain entities other than those of the {type} type. Here is the entity type information: {type}: {label_descri}

                          If there are entities:
                          -Please extract all entities other than those of the {type} type.
                          -Please rate the relevance of extracted entities to the type {type} on a scale of 1 to 10.
                          -Please only respond all extracted entities with rating strictly in the format: "Entity//Rating" for only one phrase or "Entity//Rating, Entity//Rating" for two or more phrases, without any other words.

                          If there are no corresponding entities, please respond with an empty string: "" without any other words.

                          sentence：{sentence}

                          """

            predict_entity_list = []
            predict_rating_list = []
            predict_MISC_entity_list = []
            predict_MISC_rating_list = []

            prompt1 = ChatPromptTemplate.from_template(template1)
            output_parser = CommaSeparatedListOutputParser()
            chain1 = prompt1 | llm | output_parser
            res1 = chain1.invoke({"label_descri":label_descri,"type":entity_type,"sentence": cell_sentence}).replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*", "").replace(".", "")

            print("PER:")
            print(res1)

            res1 = remove_before_last_colon(res1)

            if 'This' not in res1 and 'There' not in res1 and res1 != '':
                res1_list = [item.strip() for item in res1.split(',')]

                res1_list = [item for item in res1_list if item not in ('', []) and item is not None]

                print(res1_list)

                for predict in res1_list:
                    if "//" in predict:
                        if predict.split("//")[0] != "" and predict.split("//")[1] != "":
                            if predict.split("//")[0] not in predict_entity_list:
                                predict_entity_list.append(predict.split("//")[0])
                                predict_rating_list.append(predict.split("//")[1])

                predict_rating_list, indices_to_remove = filter_non_int_convertible_elements(
                    predict_rating_list)

                for index in reversed(indices_to_remove):
                    del predict_entity_list[index]

            prompt4 = ChatPromptTemplate.from_template(template4)
            output_parser = CommaSeparatedListOutputParser()
            chain4 = prompt4 | llm | output_parser
            res4 = chain4.invoke({"label_descri":label_descri,"type":entity_type,"sentence": cell_sentence}).replace('"', '').replace("'", "").replace("\n","").replace("* ", "").replace("*", "").replace(".", "")

            print("MISC:")
            print(res4)

            res4 = remove_before_last_colon(res4)


            if 'This' not in res4 and 'There' not in res4 and res4 != '':
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

            row_rule_list_for_wrongclass = []

            for i in range(len(predict_entity_list)):
                rule_list = []

                for p in range(len(predict_MISC_entity_list)):

                    if predict_entity_list[i] == predict_MISC_entity_list[p]:

                        is_right = False

                        if "B-"+label in cell_class:
                            indexes = []
                            for o in range(len(cell_class)):
                                if cell_class[o] == "B-"+label:
                                    indexes.append(o)

                            for j in range(len(indexes)):
                                if cell_entity[indexes[j]].lower() == predict_entity_list[i].split()[0].lower():

                                    if len(predict_entity_list[i].split()) > 1:
                                        if len(cell_entity) > indexes[j] + 1:
                                            if cell_class[indexes[j] + 1] == 'I-'+label:
                                                if cell_entity[indexes[j] + 1].lower() == \
                                                        predict_entity_list[i].split()[1].lower():
                                                    is_right = True
                                                    break


                                    elif len(cell_entity) > indexes[j] + 1:
                                        if cell_class[indexes[j] + 1] != 'I-'+label:
                                            is_right = True
                                            break

                                    else:
                                        is_right = True
                                        break

                        if is_right:
                            rule_list.append(int(predict_rating_list[i]))
                            rule_list.append(int(
                                predict_MISC_rating_list[predict_MISC_entity_list.index(predict_entity_list[i])]))
                            rule_list.append(1)
                            row_rule_list_for_wrongclass.append(rule_list)
                        else:
                            rule_list.append(int(predict_rating_list[i]))
                            rule_list.append(int(
                                predict_MISC_rating_list[predict_MISC_entity_list.index(predict_entity_list[i])]))
                            rule_list.append(0)
                            row_rule_list_for_wrongclass.append(rule_list)

                if len(rule_list) > 0:
                    print(rule_list)
                    print(row_rule_list_for_wrongclass)

            row_rule_list_for_missTarg = []

            for r in range(len(predict_MISC_entity_list)):
                rule_list = []
                if predict_MISC_entity_list[r] not in predict_entity_list:

                    is_right = False

                    if "B-"+label in cell_class:
                        indexes = []
                        for o in range(len(cell_class)):
                            if cell_class[o] == "B-"+label:
                                indexes.append(o)

                        for t in range(len(indexes)):

                            if cell_entity[indexes[t]].lower() == predict_MISC_entity_list[r].split()[0].lower():

                                if len(predict_MISC_entity_list[r].split()) > 1:
                                    if len(cell_entity) > indexes[t] + 1:
                                        if cell_class[indexes[t] + 1] == 'I-'+label:
                                            if cell_entity[indexes[t] + 1].lower() == \
                                                    predict_MISC_entity_list[r].split()[1].lower():
                                                is_right = True
                                                break


                                elif len(cell_entity) > indexes[t] + 1:
                                    if cell_class[indexes[t] + 1] != 'I-'+label:
                                        is_right = True
                                        break

                                else:
                                    is_right = True
                                    break

                    if is_right:
                        rule_list.append(0)
                        rule_list.append(int(predict_MISC_rating_list[r]))
                        rule_list.append(1)
                        row_rule_list_for_missTarg.append(rule_list)
                    else:
                        rule_list.append(0)
                        rule_list.append(int(predict_MISC_rating_list[r]))
                        rule_list.append(0)
                        row_rule_list_for_missTarg.append(rule_list)

                if len(rule_list) > 0:
                    print(rule_list)
                    print(row_rule_list_for_missTarg)

            row_rule_list_for_missMISC = []

            for r in range(len(predict_entity_list)):
                rule_list = []
                if predict_entity_list[r] not in predict_MISC_entity_list:

                    is_right = False

                    if "B-"+label in cell_class:
                        indexes = []
                        for o in range(len(cell_class)):
                            if cell_class[o] == "B-"+label:
                                indexes.append(o)

                        for t in range(len(indexes)):

                            if cell_entity[indexes[t]].lower() == predict_entity_list[r].split()[0].lower():

                                if len(predict_entity_list[r].split()) > 1:
                                    if len(cell_entity) > indexes[t] + 1:
                                        if cell_class[indexes[t] + 1] == 'I-'+label:
                                            if cell_entity[indexes[t] + 1].lower() == \
                                                    predict_entity_list[r].split()[1].lower():
                                                is_right = True
                                                break


                                elif len(cell_entity) > indexes[t] + 1:
                                    if cell_class[indexes[t] + 1] != 'I-'+label:
                                        is_right = True
                                        break

                                else:
                                    is_right = True
                                    break

                    if is_right:
                        rule_list.append(int(predict_rating_list[r]))
                        rule_list.append(0)
                        rule_list.append(1)
                        row_rule_list_for_missMISC.append(rule_list)
                    else:
                        rule_list.append(int(predict_rating_list[r]))
                        rule_list.append(0)
                        rule_list.append(0)
                        row_rule_list_for_missMISC.append(rule_list)

                if len(rule_list) > 0:
                    print(rule_list)
                    print(row_rule_list_for_missMISC)

            all_rule_list.append(row_rule_list_for_wrongclass)
            all_rule_list.append(row_rule_list_for_missTarg)
            all_rule_list.append(row_rule_list_for_missMISC)

            print("---" * 30)

            line_num += 1
        print(all_rule_list)





if __name__ == "__main__":

    workbook = openpyxl.load_workbook('../dataset/conll03_test.xlsx')

    sheet = workbook.active

    llm_list = ["qwen2:7b", "llama3:8b", "gemma:7b", "llama3.1:8b", "qwen2.5:7b", "mistral:7b", "gemma2:9b"]

    llm_onlyname_list = ["qwen2", "llama3", "gemma", "llama3.1", "qwen2.5", "mistral", "gemma2"]

    # label description
    LD = ""

    RatingCollect(llm_list, llm_onlyname_list, "Person", sheet, LD)



















