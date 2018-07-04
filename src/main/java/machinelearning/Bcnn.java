package machinelearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.*;
import java.util.stream.Collectors;

import static org.nd4j.linalg.factory.Nd4j.getExecutioner;

//import static org.junit.Assert.assertEquals;
//assertEquals(output, output_tmp);

class Bcnn{
    private INDArray embeddings;
    private HashMap w_idx;
    private ArrayList<INDArray> conv_p_w;
    private INDArray conv_p_b;
    private INDArray logreg_W;
    private INDArray logreg_b;
    private HashMap label_map;
    private Integer padding_width;
    private String model;

    Bcnn(INDArray embeddings, HashMap w_idx){
        this.embeddings = embeddings;
        this.w_idx = w_idx;
        this.model = "softmax";
    }

    final Bcnn set_convolution_model(ArrayList<INDArray> conv_p_w, INDArray conv_p_b, INDArray logreg_W, INDArray logreg_b){
        this.conv_p_w = conv_p_w;
        this.conv_p_b = conv_p_b;
        this.logreg_W = logreg_W.divi(2);
        this.logreg_b = logreg_b;

        // revert the convolution parameter because the convolution algorithm is little different compared with python
        for(INDArray conv_p_w_item : this.conv_p_w) {
            for(int i = 0; i < conv_p_w_item.shape()[0]; ++i){
                for(int j = 0; j < conv_p_w_item.shape()[1]; ++j){
                    INDArrayIndex[] index = new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all()};
                    conv_p_w_item.put(index, Nd4j.reverse(conv_p_w_item.get(index)));
                }
            }
        }

        padding_width = 0;
        for (INDArray p:conv_p_w){
            int[] w_shape = p.shape();
            int tmp_width = w_shape[w_shape.length - 2];
            if (tmp_width > padding_width)
                padding_width = tmp_width;
        }
        padding_width -= 1;

        return this;
    }

    final Bcnn setLabel_map(HashMap label_map) {
        this.label_map = label_map;
        return this;
    }

    final Bcnn setModel(String model) {
        this.model = model;
        return this;
    }

    Map<String, Double> predict(List<String> inputWords){
        if (null == embeddings || null == w_idx || null == conv_p_w || null == conv_p_b || null == logreg_W || null == logreg_b){
            return null;
        }

        try {
            inputWords = clean_str(inputWords);
            INDArray sent_matrix = get_sentence_matrix(inputWords);
            INDArray conv_input = get_convolution_input(sent_matrix);
            INDArray outputs = convolution(conv_input);
            INDArray logic_reg = logic_regression(outputs);
            INDArray output;
            switch (model) {
                case "softmax":
                    output = softmax(logic_reg);
                    break;
                case "topk":
                    output = topk(logic_reg);
                    break;
                default:
                    return null;
            }

            Map<String, Double> result = new HashMap<>();
            HashMap label_map_l = label_map;
            int length_input = output.length();
            for(Integer i = 0; i < length_input; ++i){
                result.put((String) label_map_l.get(i.toString()), output.getDouble(i));
            }
            return result;
        }
        catch(Exception e){
            System.out.println(e.toString());
            return null;
        }
    }

    INDArray get_convolution_input(INDArray sent_matrix) throws Exception {
            return sent_matrix;
    }

    private List<String> clean_str(List<String> inputWords){
        String sentence = "";
        for (String word: inputWords){
            sentence += word + " ";
        }

        sentence = sentence.replaceAll("[^A-Za-z0-9(),!?\'`]", " ");
        sentence = sentence.replaceAll("\'s", " \'s");
        sentence = sentence.replaceAll("\'ve", " \'ve");
        sentence = sentence.replaceAll("n\'t", " n\'t");
        sentence = sentence.replaceAll("\'re", " \'re");
        sentence = sentence.replaceAll("\'d", " \'d");
        sentence = sentence.replaceAll("\'ll", " \'ll");
        sentence = sentence.replaceAll(",", " , ");
        sentence = sentence.replaceAll("!", " ! ");
        sentence = sentence.replaceAll("\\(", " ( ");
        sentence = sentence.replaceAll("\\)", " ) ");
        sentence = sentence.replaceAll("\\?", " ? ");
        sentence = sentence.replaceAll("\\s{2,}", " ");

        return Arrays.asList(sentence.trim().split(" "));

    }

    private INDArray get_sentence_matrix(List<String> inputWords) throws Exception {
        int words_length = inputWords.size();
        int sentence_matrix_len = words_length + padding_width * 2;
        ArrayList<Double> sentence_array_list = new ArrayList<>();
        INDArray sentence_array = Nd4j.zeros(sentence_matrix_len);
        HashMap w_idx_l = w_idx;
        sentence_array_list.addAll(inputWords.stream().filter(w_idx_l::containsKey).map(ss -> (Double) w_idx_l.get(ss)).collect(Collectors.toList()));

        if (sentence_array_list.size() == 0)
            throw new RuntimeException("The inputWords have no valued words");

        int i = 0;
        for (Double value: sentence_array_list){
            sentence_array.putScalar(i + padding_width, value);
            ++i;
        }

        INDArray embeddings_l = embeddings;
        INDArray sent_matrix = Nd4j.create(sentence_matrix_len, embeddings_l.shape()[1]);
        for (i = 0; i < sentence_matrix_len; ++i){
            sent_matrix.putRow(i, embeddings_l.getRow(sentence_array.getInt(i)));
        }
        return sent_matrix;
    }

    private INDArray convolution(INDArray conv_input)throws Exception {
        conv_input = conv_input.reshape(1,1,conv_input.shape()[0],conv_input.shape()[1]);
        INDArray outputs = Nd4j.create(conv_p_w.size(), conv_p_w.get(0).shape()[0]);
        int conv_p_w_size = conv_p_w.size();
        for (int i = 0; i < conv_p_w_size; ++i){
            INDArray con_result = CommonPort.convolution(conv_input, conv_p_w.get(i), new int[]{1, 1}, new int[]{0, 0}/*, Convolution.Type.VALID*/).permute(1, 0);
            INDArray conv_p_b_i = conv_p_b.get(NDArrayIndex.point(i), NDArrayIndex.all()).permute(1, 0);
            INDArray conv_out_relu = con_result.addColumnVector(conv_p_b_i).max(1);
            outputs.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all()}, conv_out_relu);
        }
        INDArray output = Nd4j.toFlattened(outputs);
        BooleanIndexing.replaceWhere(output, 0.0, Conditions.lessThan(0.0));
        return output;
    }

    private INDArray logic_regression(INDArray input){
        assert(input.shape()[0] == 300);
        return input.mmul(logreg_W).add(logreg_b);
    }

    private INDArray softmax(INDArray input){
        getExecutioner().exec(new Exp(input));
        double sum = input.sumNumber().doubleValue();

        //getExecutioner().execAndReturn(new IAMax(input)).getFinalResult();
        return input.divi(sum);
    }

    private INDArray topk(INDArray input){
        getExecutioner().exec(new Sigmoid(input));
        return input;
    }
}