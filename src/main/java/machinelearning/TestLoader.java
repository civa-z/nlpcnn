package machinelearning;

import javafx.util.Pair;

import java.io.*;
import java.util.ArrayList;

class TestLoader extends FileLoader {
    ArrayList<Pair<String, String>> getTest_map() {
        return test_map;
    }

    private ArrayList<Pair<String, String>> test_map;

    TestLoader(String filename) {
        super(filename);
        test_map = new ArrayList<>();
    }

    boolean analysis() throws IOException{
        ArrayList<String> string_list = load();
        for (String aLine:string_list){
            String[] specimen = aLine.trim().split("\t");
            if (2 != specimen.length)
                continue;
            test_map.add(new Pair<>(specimen[0], specimen[1]));
        }
        return true;
    }
}