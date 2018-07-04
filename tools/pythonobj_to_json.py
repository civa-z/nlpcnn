import cPickle
import json
import sys
import os

if __name__ == "__main__":
    filename = sys.argv[1]

    if not os.path.exists(filename):
        print "Input file not exist"
        exit()

    input_data = cPickle.load(open(filename, "rb"))
    output_data = []
    if len(input_data) != 6 and len(input_data) != 7:
        print "Data analysis error"

    w2i, U, conv_params, LR_W, LR_b, label_map = input_data[0], input_data[1], input_data[2], input_data[3], input_data[4], input_data[5]



    U = [[(float)(colum) for colum in row] for row in U]
    conv_params_w = [[[[[(float)(forth) for forth in third] for third in second] for second in first] for first in conv_params[i]]for i in range(0, len(conv_params), 2)]
    conv_params_b = [[(float)(first) for first in conv_params[i + 1]] for i in range(0, len(conv_params), 2)]
    LR_W = [[(float)(column) for column in row] for row in LR_W]
    LR_b = [(float)(row) for row in LR_b]

    output_data.append(w2i)
    output_data.append(U)
    output_data.append(conv_params_w)
    output_data.append(conv_params_b)
    output_data.append(LR_W)
    output_data.append(LR_b)
    output_data.append(label_map)

    if len(input_data) == 7:
        att_data = input_data[6]
        W_all, W_t, v_att, b_att= att_data[0], att_data[1], att_data[2], att_data[3]
        W_all = [[(float)(second) for second in first] for first in W_all]
        W_t = [[(float)(second) for second in first] for first in W_t]
        v_att = [(float)(first) for first in v_att]
        b_att = (float)(b_att[0])
        output_data.append(W_all)
        output_data.append(W_t)
        output_data.append(v_att)
        output_data.append(b_att)

    with open(filename + ".json", "wb") as f:
        json.dump(output_data, f)

