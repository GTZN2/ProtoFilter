import numpy as np
import io

"""
wrongclass:[>1,n,bool]
missPER:[0,n,0]
missMISC:[n,0,bool]
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fliter_for_wrongclass(rating_list):
    rating_list_for_wrongclass = []

    for i in rating_list:
        element = []
        if i[0] != 0 and i[1] != 0 and i[0] <= 10 and i[1] <= 10:
            element.append(i[0])
            element.append(i[1])
            element.append(i[2])
            rating_list_for_wrongclass.append(element)

    return rating_list_for_wrongclass


def fliter_for_missTarget(rating_list):
    rating_list_for_miss = []
    for i in rating_list:
        element = []
        if i[0] == 0 and i[0] <= 10 and i[1] <= 10:
            element.append(i[0])
            element.append(i[1])
            element.append(i[2])
            rating_list_for_miss.append(element)

    return rating_list_for_miss


def fliter_for_missMISC(rating_list):
    rating_list_for_miss = []
    for i in rating_list:
        element = []
        if i[1] == 0 and i[0] <= 10 and i[1] <= 10:
            element.append(i[0])
            element.append(i[1])
            element.append(i[2])
            rating_list_for_miss.append(element)

    return rating_list_for_miss


def fliter_for_right(rating_list):
    rating_list_for_right = []
    for i in rating_list:
        if i[-1] == 1:
            element = []
            element.append(i[0])
            element.append(i[1])
            rating_list_for_right.append(element)

    return rating_list_for_right


def fliter_for_wrong(rating_list):
    rating_list_for_wrong = []
    for i in rating_list:
        if i[-1] == 0:
            element = []
            element.append(i[0])
            element.append(i[1])
            rating_list_for_wrong.append(element)

    return rating_list_for_wrong


def calculate_prototype_and_ni(rating_list):
    data = np.array(rating_list)


    covariance_matrix = np.cov(data, rowvar=False)
    ni = np.linalg.inv(covariance_matrix)


    mean_point = np.mean(rating_list, 0)


    distances = np.linalg.norm(data - mean_point, axis=1)


    mean_distance = np.mean(distances)

    dispersion = sigmoid(np.trace(covariance_matrix))

    return np.mean(rating_list, 0), dispersion, mean_distance, ni


def calculate_prototype(rating_list):
    data = np.array(rating_list)


    covariance_matrix = np.cov(data, rowvar=False)


    mean_point = np.mean(rating_list, 0)


    distances = np.linalg.norm(data - mean_point, axis=1)

    mean_distance = np.mean(distances)

    dispersion = sigmoid(np.trace(covariance_matrix))

    return np.mean(rating_list, 0), dispersion, mean_distance


def normList(raw_list):
    normls = []
    for i in raw_list:
        if len(i) == 1 and 0 <= i[0][0] <= 10 and 0 <= i[0][1] <= 10:
            normls.append(i[0])
        elif len(i) > 1:
            temp_list_right = []
            temp_list_wrong = []
            # for j in i:
            #     if j[0]<=10 and j[1]<=10:
            #         if j[-1] == 1:
            #             temp_list_right.append(j)
            #         else:
            #             temp_list_wrong.append(j)
            # if len(temp_list_right)>1:
            #     normls.append(np.mean(temp_list_right, 0).tolist())
            # if len(temp_list_wrong)>1:
            #     normls.append(np.mean(temp_list_wrong, 0).tolist())

            for j in i:
                if 0 <= j[0] <= 10 and 0 <= j[1] <= 10:
                    normls.append(j)

    return normls


# 创建一个 StringIO 对象
output_buffer = io.StringIO()

raw_list = []

normls = normList(raw_list)

rating_list_for_wrongclass = fliter_for_wrongclass(normls)
rating_list_for_missTarget = fliter_for_missTarget(normls)
rating_list_for_missMISC = fliter_for_missMISC(normls)

rating_list_for_wrongclass_right = fliter_for_right(rating_list_for_wrongclass)
rating_list_for_wrongclass_wrong = fliter_for_wrong(rating_list_for_wrongclass)

rating_list_for_missTarget_right = fliter_for_right(rating_list_for_missTarget)
rating_list_for_missTarget_wrong = fliter_for_wrong(rating_list_for_missTarget)

rating_list_for_missMISC_right = fliter_for_right(rating_list_for_missMISC)
rating_list_for_missMISC_wrong = fliter_for_wrong(rating_list_for_missMISC)

prototype_for_wrongclass_right, prototype_for_wrongclass_right_disp, prototype_for_wrongclass_right_meandist, prototype_for_wrongclass_right_ni = calculate_prototype_and_ni(
    rating_list_for_wrongclass_right)
prototype_for_wrongclass_wrong, prototype_for_wrongclass_wrong_disp, prototype_for_wrongclass_wrong_meandist, prototype_for_wrongclass_wrong_ni = calculate_prototype_and_ni(
    rating_list_for_wrongclass_wrong)

prototype_for_missTarget_right, prototype_for_missTarget_right_disp, prototype_for_missTarget_right_meandist = calculate_prototype(
    rating_list_for_missTarget_right)
prototype_for_missTarget_wrong, prototype_for_missTarget_wrong_disp, prototype_for_missTarget_wrong_meandist = calculate_prototype(
    rating_list_for_missTarget_wrong)

prototype_for_missMISC_right, prototype_for_missMISC_right_disp, prototype_for_missMISC_right_meandist = calculate_prototype(
    rating_list_for_missMISC_right)
prototype_for_missMISC_wrong, prototype_for_missMISC_wrong_disp, prototype_for_missMISC_wrong_meandist = calculate_prototype(
    rating_list_for_missMISC_wrong)

print("rating_list_for_wrongclass_right: " + str(rating_list_for_wrongclass_right) + '\n')
print("prototype_for_wrongclass_right: " + str(prototype_for_wrongclass_right) + '\n')
print("prototype_for_wrongclass_right_disp: " + str(prototype_for_wrongclass_right_disp) + '\n')
print("prototype_for_wrongclass_right_meandist: " + str(prototype_for_wrongclass_right_meandist) + '\n')
print("prototype_for_wrongclass_right_ni: " + str(prototype_for_wrongclass_right_ni) + '\n')

print("rating_list_for_wrongclass_wrong: " + str(rating_list_for_wrongclass_wrong) + '\n')
print("prototype_for_wrongclass_wrong: " + str(prototype_for_wrongclass_wrong) + '\n')
print("prototype_for_wrongclass_wrong_disp: " + str(prototype_for_wrongclass_wrong_disp) + '\n')
print("prototype_for_wrongclass_wrong_meandist: " + str(prototype_for_wrongclass_wrong_meandist) + '\n')
print("prototype_for_wrongclass_wrong_ni: " + str(prototype_for_wrongclass_wrong_ni) + '\n')

print("-----" * 100)
print("rating_list_for_missTarget_right: " + str(rating_list_for_missTarget_right) + '\n')
print("prototype_for_missTarget_right: " + str(prototype_for_missTarget_right) + '\n')
print("prototype_for_missTarget_right_disp: " + str(prototype_for_missTarget_right_disp) + '\n')
# print("prototype_for_missTarget_right_meandist: "+str(prototype_for_missTarget_right_meandist)+'\n')


print("rating_list_for_missTarget_wrong: " + str(rating_list_for_missTarget_wrong) + '\n')
print("prototype_for_missTarget_wrong: " + str(prototype_for_missTarget_wrong) + '\n')
print("prototype_for_missTarget_wrong_disp: " + str(prototype_for_missTarget_wrong_disp) + '\n')
# print("prototype_for_missTarget_wrong_meandist: "+str(prototype_for_missTarget_wrong_meandist)+'\n')

print("-----" * 100)
# 如果有miss，则靠近prototype_for_miss_wrong：MISC→PER
print("rating_list_for_missMISC_right: " + str(rating_list_for_missMISC_right) + '\n')
print("prototype_for_missMISC_right: " + str(prototype_for_missMISC_right) + '\n')
print("prototype_for_missMISC_right_disp: " + str(prototype_for_missMISC_right_disp) + '\n')
print("prototype_for_missMISC_right_meandist: " + str(prototype_for_missMISC_right_meandist) + '\n')

print("rating_list_for_missMISC_wrong: " + str(rating_list_for_missMISC_wrong) + '\n')
print("prototype_for_missMISC_wrong: " + str(prototype_for_missMISC_wrong) + '\n')
print("prototype_for_missMISC_wrong_disp: " + str(prototype_for_missMISC_wrong_disp) + '\n')
print("prototype_for_missMISC_wrong_meandist: " + str(prototype_for_missMISC_wrong_meandist) + '\n')

print("rating_list_for_wrongclass_right: " + str(rating_list_for_wrongclass_right) + '\n', file=output_buffer)

print("prototype_for_wrongclass_right: " + str(prototype_for_wrongclass_right) + '\n', file=output_buffer)
print("prototype_for_wrongclass_right_disp: " + str(prototype_for_wrongclass_right_disp) + '\n', file=output_buffer)
print("rating_list_for_wrongclass_wrong: " + str(rating_list_for_wrongclass_wrong) + '\n', file=output_buffer)
#
print("prototype_for_wrongclass_wrong: " + str(prototype_for_wrongclass_wrong) + '\n', file=output_buffer)
print("prototype_for_wrongclass_wrong_disp: " + str(prototype_for_wrongclass_wrong_disp) + '\n', file=output_buffer)

print("-----" * 100, file=output_buffer)
#MISC→PER
print("rating_list_for_missTarget_right: " + str(rating_list_for_missTarget_right) + '\n', file=output_buffer)
print("prototype_for_missTarget_right: " + str(prototype_for_missTarget_right) + '\n', file=output_buffer)
print("prototype_for_missTarget_right_disp: " + str(prototype_for_missTarget_right_disp) + '\n', file=output_buffer)
print("rating_list_for_missTarget_wrong: " + str(rating_list_for_missTarget_wrong) + '\n', file=output_buffer)
print("prototype_for_missTarget_wrong: " + str(prototype_for_missTarget_wrong) + '\n', file=output_buffer)
print("prototype_for_missTarget_wrong_disp: " + str(prototype_for_missTarget_wrong_disp) + '\n', file=output_buffer)

print("-----" * 100, file=output_buffer)
# 如果有miss，则靠近prototype_for_miss_wrong：MISC→PER
print("rating_list_for_missMISC_right: " + str(rating_list_for_missMISC_right) + '\n', file=output_buffer)
print("prototype_for_missMISC_right: " + str(prototype_for_missMISC_right) + '\n', file=output_buffer)
print("prototype_for_missTarget_right_disp: " + str(prototype_for_missTarget_right_disp) + '\n', file=output_buffer)
print("rating_list_for_missMISC_wrong: " + str(rating_list_for_missMISC_wrong) + '\n', file=output_buffer)
print("prototype_for_missMISC_wrong: " + str(prototype_for_missMISC_wrong) + '\n', file=output_buffer)
print("prototype_for_missTarget_wrong_disp: " + str(prototype_for_missTarget_wrong_disp) + '\n', file=output_buffer)

output_str = output_buffer.getvalue()

# 将内容写入文件
# with open(r"C:\NER\results\LOC_Proto\CONLL03_testset_"+"qwen2"+"_LOC_Proto.txt", "w", encoding="utf-8") as file:
# with open(r"C:\NER\results\ORG_Proto\CONLL03_testset_" + "qwen2" + "_ORG_Proto.txt", "w", encoding="utf-8") as file:
#     file.write(output_str)

# 关闭 StringIO 对象
output_buffer.close()