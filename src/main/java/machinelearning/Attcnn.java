package machinelearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.HashMap;

import static org.nd4j.linalg.factory.Nd4j.getExecutioner;

//import static org.junit.Assert.assertEquals;
//assertEquals(output, output_tmp);


class Attcnn extends Bcnn{
    private INDArray W_all;
    private INDArray W_t;
    private INDArray v_att;
    private double b_att;
    private double _dist_lambda;

    Attcnn(INDArray embeddings, HashMap w_idx) {
        super(embeddings, w_idx);
    }

    final Attcnn set_attention_mode(INDArray W_all, INDArray W_t, INDArray v_att, double b_att, double dist_lambda){
        this.W_all =W_all;
        this.W_t = W_t;
        this.v_att = v_att;
        this.b_att = b_att;
        this._dist_lambda = dist_lambda;
        return this;
    }

    @Override
    INDArray get_convolution_input(INDArray sent_matrix) throws Exception{
        INDArray attention_mode = calc_att_noscan_dist(sent_matrix);
        return Nd4j.hstack(sent_matrix, attention_mode);
    }

    private INDArray calc_att_noscan_dist(INDArray input) throws Exception{
        INDArray _input_repeated_proj = input.mmul(W_t).dup();
        _input_repeated_proj = _input_repeated_proj.reshape(1, _input_repeated_proj.shape()[0], _input_repeated_proj.shape()[1]);

        INDArray w_all_proj = input.mmul(W_all).dup();
        w_all_proj = w_all_proj.reshape(w_all_proj.shape()[0], 1, w_all_proj.shape()[1]);

        _input_repeated_proj = CommonPort.repeat_ndarray(_input_repeated_proj, 0,  w_all_proj.shape()[0]);
        //_input_repeated_proj = _input_repeated_proj.repeat(0, new int[]{w_all_proj.shape()[0],1,1});

        w_all_proj = CommonPort.repeat_ndarray(w_all_proj, 1, _input_repeated_proj.shape()[1]);
        INDArray pctx = _input_repeated_proj.add(w_all_proj);

        getExecutioner().exec(new Tanh(pctx));
        INDArray pctx_tmp = pctx.reshape(pctx.shape()[0] * pctx.shape()[1], pctx.shape()[2]);
        INDArray alpha = pctx_tmp.mmul(v_att).add(b_att).reshape(pctx.shape()[0], pctx.shape()[1]);
        getExecutioner().exec(new Exp(alpha));
        BooleanIndexing.replaceWhere(alpha, 0, Conditions.equals(1.0));

        INDArray disk_mask_matrix = get_dist_mask_matrix(input.shape()[0]);
        if (1 == _dist_lambda){
            alpha.muli(disk_mask_matrix);
        }

        INDArray mask_matrix_value = Nd4j.diag(Nd4j.ones(input.shape()[0])).rsub(1);
        alpha.muli(mask_matrix_value);

        INDArray alpha_sum = alpha.sum(0);
        BooleanIndexing.replaceWhere(alpha_sum, 1, Conditions.equals(0));
        alpha.diviRowVector(alpha_sum);

        return alpha.permute(1, 0).mmul(input).dup();
    }

    private INDArray get_dist_mask_matrix(int width) throws Exception{
        INDArray dist_mask;
        if (1 == _dist_lambda)
            dist_mask = Nd4j.ones(width, width);
        else{
            dist_mask = Nd4j.zeros(width, width);
            INDArray diag_a = Nd4j.ones(width);
            for(int i = 0; i < width; ++i ){
                dist_mask.add(Nd4j.diag(diag_a, i).mul(i + 1));
            }
            dist_mask.addi(dist_mask.transpose()).sub(dist_mask.mmul(Nd4j.diag(diag_a)));
        }
        INDArray dist_lambda_matrix = Nd4j.ones(width, width).mul(_dist_lambda);
        dist_mask = CommonPort.pow(dist_lambda_matrix, dist_mask);
        return dist_mask;
    }
}
