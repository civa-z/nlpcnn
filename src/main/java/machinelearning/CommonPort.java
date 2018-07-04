package machinelearning;

import com.google.gson.JsonElement;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;

class CommonPort{

    static INDArray load_json_to_NDArray(JsonElement json_array, int depth) {
        if (0 >= depth)
            return null;

        ArrayList<Integer> shape = get_shape(json_array, depth);
        Integer[] shape_a = new Integer[shape.size()];
        shape.toArray(shape_a);
        int[] shape_int = ArrayUtils.toPrimitive(shape_a);

        INDArray result;
        if (1 == shape_int.length)
            result = Nd4j.create(shape_int[0]);
        else
            result = Nd4j.zeros(shape_int);

        int json_array_size = json_array.getAsJsonArray().size();
        for(int i = 0; i < json_array_size; ++i){
            if (1 == depth)
                result.putScalar(i, json_array.getAsJsonArray().get(i).getAsDouble());
            else
                result.putRow(i, load_json_to_NDArray(json_array.getAsJsonArray().get(i), depth - 1));
        }

        if (1 == shape_int.length){
            result = result.permute(1, 0);
        }

        return result;
    }

    static private ArrayList<Integer> get_shape(JsonElement json_array, int depth) {
        ArrayList<Integer> result = new ArrayList<>();

        if (1 == depth)
            result.add(json_array.getAsJsonArray().size());
        else if (1 < depth){
            result.add(json_array.getAsJsonArray().size());
            result.addAll(get_shape(json_array.getAsJsonArray().get(0), depth - 1));
        }
        return result;
    }

    static INDArray pow(INDArray origin, INDArray pow) throws Exception{
        if (!Arrays.equals(origin.shape(), pow.shape())){
            throw new RuntimeException(new NoAvailableShapeException("The shape of origin and pow is not equal"));
        }
        if (2 != origin.shape().length){
            throw new RuntimeException(new NoAvailableShapeException("The origin is not a 2D matrix"));
        }

        INDArray output = Nd4j.zeros(origin.shape()[0], origin.shape()[1]);
        int origin_shape_0 = origin.shape()[0];
        int origin_shape_1 = origin.shape()[1];

        int i, j;
        for(i = 0; i < origin_shape_0; ++i){
            for(j = 0; j < origin_shape_1; ++j){
                output.put(i, j, Math.pow(origin.getDouble(i,j), pow.getDouble(i, j)));
            }
        }
        return output;
    }

    /*
    This funciton only support repeat the dimension of which the length of it is one.
    * */
    static INDArray repeat_ndarray(INDArray input, int dimension, int repeats){
        if (1 != input.shape()[dimension]){
            return input;
        }

        int[] permute_dimension = input.shape().clone();
        for (int i = 0; i < permute_dimension.length; ++i)
            permute_dimension[i] = i;
        permute_dimension[0] = dimension;
        permute_dimension[dimension] = 0;
        INDArray input_dup_permute = input.permute(permute_dimension);

        int[] shape_tmp = input.shape();
        shape_tmp[dimension] = shape_tmp[0];
        shape_tmp[0] = repeats;
        INDArray array_tmp = Nd4j.create(shape_tmp);

        for (int i = 0; i < repeats; ++ i){
            array_tmp.putRow(i, input_dup_permute.getRow(0));
        }

        return array_tmp.permute(permute_dimension);
    }

    private static class NoAvailableShapeException extends Exception {
        NoAvailableShapeException(String s) {
            super(s);
        }
    }

    static INDArray convolution(INDArray input, INDArray kernel, int strides[], int pad[] /*Convolution.Type type*/)throws Exception{
        int Batch = input.shape()[0];
        int inDepth = input.shape()[1];
        int height = input.shape()[2];
        int width = input.shape()[3];

        int outDepth = kernel.shape()[0];
        //int nChannelsIn = kernel.shape()[1];
        int kernel_H = kernel.shape()[2];
        int kernel_W = kernel.shape()[3];

        int outH = Convolution.outSize(height, kernel_H, strides[0], pad[0],false);
        int outW = Convolution.outSize(width, kernel_W, strides[1], pad[1], false);

        INDArray col = Nd4j.create(new int[]{Batch, outH, outW, inDepth, kernel_H, kernel_W},'c');
        INDArray col2 = col.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(input, kernel_H, kernel_W, strides[0], strides[1], pad[0], pad[1], false, col2);
        INDArray reshapedCol = Shape.newShapeNoCopy(col,new int[]{Batch * outH * outW, inDepth * kernel_H * kernel_W},false);
        INDArray permutedW = kernel.permute(3, 2, 1, 0);
        INDArray reshapedW = permutedW.reshape('f', kernel_H * kernel_W * inDepth, outDepth);
        return reshapedCol.mmul(reshapedW);
    }
}