package machinelearning;

import java.io.*;
import java.util.ArrayList;

abstract class FileLoader {
    private String filename;

    FileLoader(String filename){
        this.filename = filename;
    }

    ArrayList<String> load() throws IOException {
        if (null == filename)
            return null;

        File file = new File(filename);
        System.out.println("Input file " + filename + ". len :" + file.length());
        ArrayList<String> string_list = new ArrayList<>();

        BufferedReader bufR = null;
        try {
            bufR = new BufferedReader(new FileReader(file));
            String aLine;
            while (null != (aLine = bufR.readLine())) {
                string_list.add(aLine);
            }
        } finally {
            if (null != bufR)
                bufR.close();
        }

        return string_list;
    }

    boolean analysis() throws Exception {
        return true;
    }

}