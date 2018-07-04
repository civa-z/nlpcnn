package machinelearning;

import java.io.IOException;
import java.util.ArrayList;

class CnnConfiguration extends FileLoader{
    private String filename;
    private String model;
    CnnConfiguration(String filename) {
        super(filename);
    }

    boolean analysis() throws IOException {
        ArrayList<String> string_list = load();
        for (String line: string_list){
            line = line.trim();
            if (line.startsWith("#"))
                continue;

            String[] items = line.split("=");
            if (2 != items.length)
                continue;

            switch (items[0]){
                case "filename":
                    filename = items[1];
                    break;
                case "model":
                    model = items[1];
                    break;
                default:
                    break;
            }
        }
        return true;
    }

    String getFilename() {
        return filename;
    }

    String getModel() {
        return model;
    }
}
