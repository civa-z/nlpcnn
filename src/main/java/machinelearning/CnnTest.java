package machinelearning;

import javafx.util.Pair;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.collections.CollectionUtils;

import java.io.IOException;
import java.util.*;

public class CnnTest
{
    public static void main( String[] args ) throws IOException {
        String configuration = "NULL";
        String test = "NULL";
        String model = "softmax";
        //Runtime.getRuntime().gc();
        long initm=Runtime.getRuntime().totalMemory();

        try{
            CommandLineParser parser = new DefaultParser( );
            Options options = new Options( );
            options.addOption("c", "configuration", true, "configuration" );
            options.addOption("m", "model", true, "model");
            options.addOption("t", "test", true, "test");
            CommandLine commandline = parser.parse(options, args);
            configuration = commandline.getOptionValue("c");
            model = commandline.getOptionValue("m");
            test = commandline.getOptionValue("t");
        }catch(Exception e){
            e.printStackTrace();
        }

        TestLoader testloader = new TestLoader(test);
        if (!testloader.analysis()) return;

        CnnHandler cnn_handler = new CnnHandler();

        try {
            cnn_handler.initialize(configuration);
        } catch (IOException e) {
            e.printStackTrace();
            return;

        }

        List<Pair<String, String>> test_map = testloader.getTest_map();
        List<Pair<String, Map>> test_result = new ArrayList<>();
        List<List<Long>> monitor_time_list = new ArrayList<>();

        int errors = 0;
        for (Pair<String, String> test_item: test_map){
            List<Long> mt = new ArrayList<>();
            mt.add(System.nanoTime());
            String sentence = test_item.getValue();
            String[] words = sentence.split(" ");
            List<String> word_list = new ArrayList<>();
            Collections.addAll(word_list, words);
            Map<String, Double> res = cnn_handler.analyze(word_list);
            if (null == res){
                errors++;
                continue;
            }
            test_result.add(new Pair<>(test_item.getKey(), res));
            mt.add(System.nanoTime());
            monitor_time_list.add(mt);
        }

        cnn_handler.terminate();

        float average_time = get_average_time(monitor_time_list);


        //Runtime.getRuntime().gc();
        long endm=Runtime.getRuntime().totalMemory();
        for (Pair<String, Map> result:test_result){
            System.out.println("GroundTruth:" + result.getKey() + ",\tPredict:" + result.getValue());
        }
        System.out.println("\nAverage memory used: " + (endm - initm)/1000/1000 + "M");
        System.out.println("Average time used: " + average_time + "ms");
        System.out.println("Test count: " + test_result.size());
        System.out.println("Errors: " + errors);

        if (model.equals("softmax")) {
            // Used for sofrmax model
            get_hit_probability(test_result);
        }
        else if(model.equals("topk")){
            // Used for muti label
            get_hit_probability(test_result, 0.5);
        }
    }

    private static float get_hit_probability(List<Pair<String, Map>> test_result) {
        int hit_sum = 0;
        for (Pair<String, Map> specimen: test_result){
            Map<String, Double> result = (Map<String, Double>)specimen.getValue();
            Pair<String, Double>[] result_list = new Pair[result.size()];
            int i = 0;
            for(Map.Entry<String, Double> entry : result.entrySet()){
                result_list[i++] = new Pair(entry.getKey(), entry.getValue());
            }

            Comparator cmp = new MyComparator();
            Arrays.sort(result_list, cmp);

            if (specimen.getKey().equals(result_list[0].getKey())) {
                hit_sum++;
            }
        }
        float hit_probability = (float)hit_sum / test_result.size();
        System.out.println("Accuracy: " + (hit_probability * 100) + "%");
        return hit_probability;
    }

    private static float get_hit_probability(List<Pair<String, Map>> test_result, double threshold) {
        int hit_sum = 0;
        int total_num = 0;
        int item_hit_count = 0;
        int item_conut = 0;
        for (Pair<String, Map> specimen: test_result){
            Map<String, Double> result = (Map<String, Double>)specimen.getValue();

            Set<String> tem_s = new HashSet();
            CollectionUtils.addAll(tem_s, specimen.getKey().split(" "));

            boolean item_hit = true;
            for (Map.Entry<String, Double> entry_set: result.entrySet()){
                if (entry_set.getValue() > threshold && tem_s.contains(entry_set.getKey()))
                    hit_sum++;
                else if (entry_set.getValue() <= threshold && !tem_s.contains(entry_set.getKey()))
                    hit_sum++;
                else
                    item_hit =  false;
                total_num++;
            }
            item_conut++;
            if (item_hit){
                item_hit_count++;
            }
        }

        float hit_probability = (float)hit_sum / total_num;
        float hit_probability_item = (float) item_hit_count/ item_conut;
        System.out.println("Accuracy Point: " + (hit_probability * 100) + "%");
        System.out.println("Accuracy Item: " + (hit_probability_item * 100) + "%");
        return hit_probability;
    }

    private static float get_average_time(List<List<Long>> monitor_time_list) {
        //Switch the time stamp from ns ot us.
        long sum  = 0;
        for (List<Long> mt: monitor_time_list){
            Long time_previous = 0L;
            if(0 < mt.size()) {
                time_previous = mt.get(0);
            }
            for(int j = 1; j < mt.size(); ++j){
                Long time_spent = mt.get(j) - time_previous;
                time_previous = mt.get(j);
                mt.set(j, time_spent);
                //System.out.print("\t" + mt.get(j) / 1000);
            }
            mt.remove(0);
            //System.out.println();
        }

        for (List<Long> item: monitor_time_list){
            for (Long interval: item) {
                sum += (interval / 1000);
            }
        }

        return  (float) sum/monitor_time_list.size() / 1000;
    }

    private static class MyComparator implements Comparator<Pair<Integer, Double>>{

        @Override
        public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
            if(o1.getValue() > o2.getValue())
                return -1;
            else if (o1.getValue() < o2.getValue())
                return 1;
            else
                return 0;
        }
    }
}
