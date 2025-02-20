# -*- coding: utf-8 -*-

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
import io
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


def remove_think(s):
    if "</think>" in s:
        return s.split("</think>")[-1]
    else:
        return s


def NER(llm_list, llm_onlyname_list, entity_type,sheet):
    label = entity_type[:3].upper()
    for llm_index in range(len(llm_list)):
        llm = ChatOllama(model=llm_list[llm_index])
        line_num = 1

        # num. of correctly recogized entities
        llm_entity_right_num = 0

        # num. of all recogized entities
        llm_entity_num = 0

        # num. of type misidentification entities
        wrong_class_by_lmm = 0

        # num. of missed entities
        miss_by_lmm = 0

        # num. of incorrectly recognized left boundary of entities
        inside_wrong_by_lmm = 0

        # num. of incorrectly recognized right boundary of entities
        head_wrong_by_lmm = 0

        # num. of ground truth entities
        entity_num = 0

        output_buffer = io.StringIO()

        for row in sheet.iter_rows(min_row=2, values_only=True):

            print("line:" + str(line_num))

            # repeatedly recorded entities
            increase_head_wrong_by_lmm1 = 0
            increase_head_wrong_by_lmm2 = 0

            print("line:" + str(line_num), file=output_buffer)

            # sentence processing
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


            # 创建提示模板
            template = """
            The following sentence may exist entities of the type {type}.

            -If there are entities, please extract entities of the type {type} and only respond the extracted entities in the format: "Entity" for only one entity or "Entity, Entity" for two or more entities, without any other words.
            -If there are no corresponding entities, please respond with an empty string: "" without any other words.

            sentence：{sentence}
            """
            prompt = ChatPromptTemplate.from_template(template)
            output_parser = CommaSeparatedListOutputParser()
            chain = prompt | llm | output_parser
            res1 = chain.invoke({"type": entity_type,"sentence": cell_sentence}).replace('"', '').replace("'", "").replace("\n","").replace("* ","").replace("*", "").replace(".", "")


            res1 = remove_before_last_colon(res1)
            res1 = remove_think(res1)


            if 'This ' not in res1 and 'There ' not in res1 and res1 != '' and ' no ' not in res1:
                res1_list = [item.strip() for item in res1.split(',')]

                res1_list = [item for item in res1_list if item not in ('', []) and item is not None]

                print(res1_list, file=output_buffer)
                print(res1_list)

                llm_entity_num += len(res1_list)

                if "B-"+label in cell_class:
                    indexes = []
                    cell_entity_head = []
                    for i in range(len(cell_class)):
                        if cell_class[i] == "B-"+label:
                            entity_num += 1
                            indexes.append(i)
                            cell_entity_head.append(cell_entity[i])
                            if cell_entity[i].lower() not in " ".join(res1_list).lower():
                                miss_by_lmm += 1

                    for r in res1_list:
                        if r.split()[0].lower() not in [s.lower() for s in cell_entity_head]:
                            wrong_class_by_lmm += 1



                    checked_entity_in_prediction = []
                    for j in range(len(indexes)):
                        for k in range(len(res1_list)):
                            is_wrong_class = False
                            if res1_list[k] not in checked_entity_in_prediction:
                                if cell_entity[indexes[j]].lower() in " ".join(res1_list).lower():
                                    if cell_entity[indexes[j]].lower() == res1_list[k].split()[0].lower():
                                        checked_entity_in_prediction.append(res1_list[k])
                                        if len(res1_list[k].split()) > 1:
                                            if len(cell_entity) > indexes[j] + 1:
                                                if cell_class[indexes[j] + 1] == 'I-'+label:
                                                    if cell_entity[indexes[j] + 1].lower() == res1_list[k].split()[
                                                        1].lower():
                                                        llm_entity_right_num += 1
                                                        is_wrong_class = False

                                                    else:
                                                        inside_wrong_by_lmm += 1
                                                        is_wrong_class = False

                                                else:
                                                    inside_wrong_by_lmm += 1
                                                    is_wrong_class = False

                                            else:
                                                inside_wrong_by_lmm += 1
                                                is_wrong_class = False

                                        elif len(cell_entity) > indexes[j] + 1:
                                            if cell_class[indexes[j] + 1] == 'I-'+label:
                                                inside_wrong_by_lmm += 1
                                                is_wrong_class = False

                                            else:
                                                llm_entity_right_num += 1
                                                is_wrong_class = False

                                        else:
                                            llm_entity_right_num += 1
                                            is_wrong_class = False


                                    elif len(res1_list[k].split()) > 1:

                                        if cell_entity[indexes[j]].lower() == res1_list[k].split()[1].lower():

                                            if res1_list[k].split()[0].lower() not in " ".join(cell_entity).lower():
                                                increase_head_wrong_by_lmm2 += 1

                                                checked_entity_in_prediction.append(res1_list[k])

                            if is_wrong_class:
                                wrong_class_by_lmm += 1



                elif len(res1_list) > 0:
                    wrong_class_by_lmm += len(res1_list)

                if "I-"+label in cell_class:
                    indexes_inside = []
                    for i in range(len(cell_class)):
                        if cell_class[i] == "I-"+label:
                            indexes_inside.append(i)
                    for inside_entity_index in range(len(indexes_inside)):
                        if cell_entity[indexes_inside[inside_entity_index]].lower() in " ".join(res1_list).lower():
                            for entity in res1_list:
                                if cell_entity[indexes_inside[inside_entity_index]].lower() == entity.split()[
                                    0].lower():
                                    increase_head_wrong_by_lmm1 += 1
                                    if cell_entity[indexes_inside[inside_entity_index] - 1].lower() not in [
                                        s.split()[0].lower() for s in res1_list]:
                                        miss_by_lmm -= 1
                                    break
                head_wrong_by_lmm += increase_head_wrong_by_lmm1 + increase_head_wrong_by_lmm2

            elif "B-"+label in cell_class:
                for i in range(len(cell_class)):
                    if cell_class[i] == "B-"+label:
                        miss_by_lmm += 1
                        entity_num += 1
            wrong_class_by_lmm -= increase_head_wrong_by_lmm1

            print("entity_num: " + str(entity_num))
            print("llm_entity_right_num: " + str(llm_entity_right_num))
            print("llm_entity_num: " + str(llm_entity_num))
            print("wrong_class_by_lmm: " + str(wrong_class_by_lmm))
            print("inside_wrong_by_lmm: " + str(inside_wrong_by_lmm))
            print("head_wrong_by_lmm: " + str(head_wrong_by_lmm))
            print("miss_by_lmm: " + str(miss_by_lmm))
            print("---" * 30)

            print("entity_num: " + str(entity_num), file=output_buffer)
            print("llm_entity_right_num: " + str(llm_entity_right_num), file=output_buffer)
            print("llm_entity_num: " + str(llm_entity_num), file=output_buffer)
            print("wrong_class_by_lmm: " + str(wrong_class_by_lmm), file=output_buffer)
            print("inside_wrong_by_lmm: " + str(inside_wrong_by_lmm), file=output_buffer)
            print("head_wrong_by_lmm: " + str(head_wrong_by_lmm), file=output_buffer)
            print("miss_by_lmm: " + str(miss_by_lmm), file=output_buffer)
            print("---" * 30, file=output_buffer)


            line_num += 1

        output_str = output_buffer.getvalue()

        with open("result/" + llm_onlyname_list[llm_index] + "_result_no_decrib_0_shot.txt", "w", encoding="utf-8") as file:
            file.write(output_str)

        output_buffer.close()


if __name__ == "__main__":

    workbook = openpyxl.load_workbook('../dataset/conll03_test.xlsx')

    sheet = workbook.active

    llm_list = ["qwen2:7b", "llama3:8b", "gemma:7b", "llama3.1:8b", "qwen2.5:7b", "gemma2:9b", "mistral:7b", "deepseek-r1"]

    llm_onlyname_list = ["qwen2", "llama3", "gemma", "llama3.1", "qwen2.5", "gemma2", "mistral", "deepseek-r1"]

    NER(llm_list, llm_onlyname_list, "Person", sheet)



























