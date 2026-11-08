import java.io.File;
import java.util.HashMap;
import java.util.Map;
import org.scijava.Context;
import org.scijava.script.ScriptService;
public class HeadlessImageJ {
    public static void main(String[] args) throws Exception {
        System.out.println("ImageJ Java " + System.getProperty("java.version") + " (" + System.getProperty("java.vendor") + ")");
        Thread.currentThread().setContextClassLoader(HeadlessImageJ.class.getClassLoader());
        try (Context context = new Context()) {
            Map<String,Object> inputs = new HashMap<>();
            for (int i = 1; i < args.length; i += 2) inputs.put(args[i], args[i+1]);
            context.service(ScriptService.class).run(new File(args[0]), true, inputs).get();
        }
    }
}
