package machinelearning;


import java.io.IOException;
import java.util.List;
import java.util.Map;

public class CnnHandler {
    private Bcnn cnn;
    private String model;


    public CnnHandler(){
        model = "softmax";
    }

    //func: Initialization process (load of model, etc.
    final public void initialize(String configFilename) throws IOException {
        CnnConfiguration cnn_conf = new CnnConfiguration(configFilename);
        cnn_conf.analysis();
        model = cnn_conf.getModel();

        CnnModelLoader cnn_model_loader = new CnnModelLoader(cnn_conf.getFilename());
        cnn_model_loader.analysis();

        if (cnn_model_loader.isHas_attention()){
            cnn = new Attcnn(
                    cnn_model_loader.getEmbd_tuned(),
                    cnn_model_loader.getWord_idx_map())
                    .set_attention_mode(cnn_model_loader.getW_all(),
                            cnn_model_loader.getW_t(),
                            cnn_model_loader.getV_att(),
                            cnn_model_loader.getB_att(), 1)
                    .setModel(model);
        }
        else{
            cnn = new Bcnn(
                    cnn_model_loader.getEmbd_tuned(),
                    cnn_model_loader.getWord_idx_map());
        }

        cnn.setLabel_map(cnn_model_loader.getLabel_map());

        cnn.set_convolution_model(cnn_model_loader.getConv_params_w(),
                cnn_model_loader.getConv_params_b(),
                cnn_model_loader.getLR_W(),
                cnn_model_loader.getLR_b());

    }

    /*    func: Estimation of DomainGoal from input sentence
    Input: List of words, sentence
    Output: Map (Key: DomainGoal name, Value: Score)
    ex)  WEATHER-CHECK: 0.983
    SCHEDULE-CHECK: 0.001
    NEWS-CHECK: 0.001 …*/
    final public Map<String, Double> analyze(List<String> inputWords){
        return cnn.predict(inputWords);
    }

    //func: Finalization process (release of resource, etc.)
    final public void terminate(){
    }
}
