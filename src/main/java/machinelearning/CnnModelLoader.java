package machinelearning;

import com.google.gson.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


class CnnModelLoader extends FileLoader {
    private HashMap word_idx_map;
    private INDArray embd_tuned;

    private ArrayList<INDArray> conv_params_w;
    private INDArray conv_params_b;

    private INDArray LR_W;
    private INDArray LR_b;
    private HashMap label_map;

    private boolean has_attention;
    private INDArray W_all;
    private INDArray W_t;
    private INDArray v_att;
    private double b_att;

    final HashMap getWord_idx_map() {
        return word_idx_map;
    }

    final INDArray getLR_W() {
        return LR_W;
    }

    final INDArray getLR_b() {
        return LR_b;
    }

    final INDArray getEmbd_tuned() {
        return embd_tuned;
    }

    final ArrayList<INDArray> getConv_params_w() {
        return conv_params_w;
    }

    final INDArray getConv_params_b() {
        return conv_params_b;
    }

    final HashMap getLabel_map() { return label_map; }

    final boolean isHas_attention() { return has_attention; }

    final INDArray getW_all() {
        return W_all;
    }

    final INDArray getW_t() {
        return W_t;
    }

    final INDArray getV_att() {
        return v_att;
    }

    final double getB_att() {
        return b_att;
    }


    CnnModelLoader(String filename) {
        super(filename);
        has_attention = false;
    }


    final boolean analysis() throws IOException{
        ArrayList<String> string_list = load();
        if (1 != string_list.size()){
            return false;
        }
        String json_string = string_list.get(0);
        if (null == json_string)
            return false;

        JsonParser parser = new JsonParser();
        JsonElement json = parser.parse(json_string);
        JsonArray  json_cnn_model =  json.getAsJsonArray();
        Gson gson = new GsonBuilder().enableComplexMapKeySerialization().create();

        if (json_cnn_model.getAsJsonArray().size() != 7 && json_cnn_model.getAsJsonArray().size() != 11){
            System.out.println("Data analysis error");
            return false;
        }

        int index = 0;
        word_idx_map = gson.fromJson(json_cnn_model.get(index++), HashMap.class);
        embd_tuned = CommonPort.load_json_to_NDArray(json_cnn_model.get(index++), 2);
        analysis_conv_params_w(json_cnn_model.get(index++));
        conv_params_b = CommonPort.load_json_to_NDArray(json_cnn_model.get(index++), 2);
        LR_W = CommonPort.load_json_to_NDArray(json_cnn_model.get(index++), 2);
        LR_b = CommonPort.load_json_to_NDArray(json_cnn_model.get(index++), 1);
        label_map = gson.fromJson(json_cnn_model.get(index++), HashMap.class);

        if (json_cnn_model.getAsJsonArray().size() == 11){
            W_all = CommonPort.load_json_to_NDArray(json_cnn_model.get(index++), 2);
            W_t = CommonPort.load_json_to_NDArray(json_cnn_model.get(index++), 2);
            v_att = CommonPort.load_json_to_NDArray(json_cnn_model.get(index++), 1);
            b_att = json_cnn_model.get(index).getAsDouble();
            has_attention = true;
        }

        return true;
    }

    private boolean analysis_conv_params_w(JsonElement json_array){
        int size_0 = json_array.getAsJsonArray().size();
        conv_params_w = new ArrayList<>();

        for (int i = 0; i < size_0; ++i){
            INDArray parames= CommonPort.load_json_to_NDArray(json_array.getAsJsonArray().get(i), 4);
            conv_params_w.add(parames);
        }

        return true;
    }

}
