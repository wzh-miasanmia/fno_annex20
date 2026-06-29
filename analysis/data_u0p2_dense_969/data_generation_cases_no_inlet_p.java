// Simcenter STAR-CCM+ macro: data_generation_cases_no_inlet_p.java
// Read cases.csv (case_id,inlet_u,src_x,src_y) and run each case.
// Export to sessionDir/exports with a filename including all parameters.
// CAD is driven by Global Parameters (already linked in STAR).
// Macro calls cadModel.update() + updateParts + remesh.

package macro;

import java.util.*;
import java.io.*;
import java.lang.reflect.*;

import star.common.*;
import star.base.neo.*;

// CAD update
import star.cadmodeler.*;
// remesh
import star.meshing.*;

public class data_generation_cases_no_inlet_p extends StarMacro {

  private static final double U_TOL = 1.0e-12;

  private static boolean sameU(double a, double b) {
    return Math.abs(a - b) <= U_TOL;
  }

  private boolean invokeBooleanSetter(Simulation sim, Object target, String methodName, boolean value) {
    if (target == null) return false;
    try {
      Method m = target.getClass().getMethod(methodName, boolean.class);
      m.invoke(target, value);
      sim.println("Applied " + methodName + "(" + value + ") on " + target.getClass().getSimpleName());
      return true;
    } catch (Exception ignore) {
      return false;
    }
  }

  private boolean invokeAnyBooleanSetter(Simulation sim, Object target, boolean enabled, String[] methodNames) {
    for (int i = 0; i < methodNames.length; i++) {
      if (invokeBooleanSetter(sim, target, methodNames[i], enabled)) {
        return true;
      }
    }
    return false;
  }

  private String getObjectDisplayName(Object obj) {
    if (obj == null) return "";

    try {
      Method m = obj.getClass().getMethod("getPresentationName");
      Object value = m.invoke(obj);
      if (value != null) return value.toString();
    } catch (Exception ignore) {
    }

    try {
      Method m = obj.getClass().getMethod("getBeanDisplayName");
      Object value = m.invoke(obj);
      if (value != null) return value.toString();
    } catch (Exception ignore) {
    }

    return obj.getClass().getSimpleName();
  }

  private List<Object> getAllSolverObjects(Simulation sim) {
    List<Object> solvers = new ArrayList<Object>();

    try {
      Object solverManager = sim.getSolverManager();
      Method getObjects = solverManager.getClass().getMethod("getObjects");
      Object values = getObjects.invoke(solverManager);

      if (values instanceof Collection<?>) {
        for (Object solver : (Collection<?>) values) {
          if (solver != null) solvers.add(solver);
        }
      }
    } catch (Exception e) {
      sim.println("[WARN] Failed to enumerate solvers: " + e.getMessage());
    }

    return solvers;
  }

  private String normalizeSolverName(String name) {
    if (name == null) return "";
    return name.replaceAll("[^A-Za-z0-9]+", "").toLowerCase(Locale.US);
  }

  private void logAvailableSolvers(Simulation sim) {
    List<Object> solvers = getAllSolverObjects(sim);
    sim.println("[INFO] Available solvers in current simulation: " + solvers.size());
    for (Object solver : solvers) {
      sim.println("  - " + getObjectDisplayName(solver) + " [" + solver.getClass().getName() + "]");
    }
  }

  private Object getSolverByName(Simulation sim, String solverName) {
    String wanted = normalizeSolverName(solverName);
    List<Object> solvers = getAllSolverObjects(sim);

    for (Object solver : solvers) {
      String displayName = normalizeSolverName(getObjectDisplayName(solver));
      if (wanted.equals(displayName)) {
        return solver;
      }
    }

    for (Object solver : solvers) {
      String displayName = normalizeSolverName(getObjectDisplayName(solver));
      String className = normalizeSolverName(solver.getClass().getSimpleName());
      if (displayName.contains(wanted) || wanted.contains(displayName)
          || className.contains(wanted) || wanted.contains(className)) {
        sim.println("[INFO] Fuzzy matched solver '" + solverName + "' -> '" + getObjectDisplayName(solver) + "'");
        return solver;
      }
    }

    sim.println("[WARN] Solver not found by name: " + solverName);
    logAvailableSolvers(sim);
    return null;
  }

  private void setSolverEnabledByName(Simulation sim, String solverName, boolean enabled) {
    Object solver = getSolverByName(sim, solverName);
    if (solver == null) {
      sim.println("[WARN] Solver not found: " + solverName);
      return;
    }

    sim.println("[INFO] Toggling solver '" + solverName + "' on object '"
      + getObjectDisplayName(solver) + "' [" + solver.getClass().getName() + "]");

    boolean ok =
      invokeAnyBooleanSetter(sim, solver, !enabled, new String[] {"setFrozen", "setIsFrozen", "setFreeze"}) ||
      invokeAnyBooleanSetter(sim, solver, enabled, new String[] {"setEnabled", "setIsEnabled", "setActive", "setIsActive"});

    if (!ok) {
      sim.println("[WARN] Could not toggle solver '" + solverName + "'. Please check its API/name in your STAR version.");
    } else {
      sim.println("[INFO] Solver '" + solverName + "' -> enabled=" + enabled);
    }
  }

  private void enableFullFlowAndScalar(Simulation sim) {
    setSolverEnabledByName(sim, "Segregated Flow", true);
    setSolverEnabledByName(sim, "Lag K-Epsilon Turbulence", true);
    setSolverEnabledByName(sim, "K-Epsilon Turbulent Viscosity", true);
    setSolverEnabledByName(sim, "Passive Scalar", true);
  }

  private void enableScalarOnlyMode(Simulation sim) {
    setSolverEnabledByName(sim, "Segregated Flow", false);
    setSolverEnabledByName(sim, "Lag K-Epsilon Turbulence", false);
    setSolverEnabledByName(sim, "K-Epsilon Turbulent Viscosity", false);
    setSolverEnabledByName(sim, "Passive Scalar", true);
  }

  public void execute() {
    execute0();
  }

  private static class CaseRow {
    String caseId;
    double inletU;
    double srcX;
    double srcY;
  }

  private void execute0() {

    Simulation sim = getActiveSimulation();
    Solution sol = sim.getSolution();

    // ============================
    // 0) Output folder
    // ============================
    String sessionDir = sim.getSessionDir();
    sim.println("Session directory = " + sessionDir);

    File outDir = new File(sessionDir, "exports");
    if (!outDir.exists()) outDir.mkdirs();
    sim.println("CSV output folder = " + outDir.getAbsolutePath());

    // ============================
    // 1) Read cases.csv
    // ============================
    File casesFile = new File(sessionDir, "cases.csv");
    sim.println("Looking for cases.csv at: " + casesFile.getAbsolutePath());
    if (!casesFile.exists()) {
      throw new RuntimeException("cases.csv not found at: " + casesFile.getAbsolutePath());
    }

    List<CaseRow> cases = new ArrayList<>();

    try (BufferedReader br = new BufferedReader(new FileReader(casesFile))) {
      String line;
      boolean first = true;
      while ((line = br.readLine()) != null) {
        line = line.trim();
        if (line.isEmpty()) continue;

        if (first) { // header
          first = false;
          continue;
        }

        // case_id,inlet_u,src_x,src_y
        String[] parts = line.split(",");
        if (parts.length < 4) {
          throw new RuntimeException("Bad line in cases.csv (need 4 columns): " + line);
        }

        CaseRow r = new CaseRow();
        r.caseId = parts[0].trim();
        r.inletU = Double.parseDouble(parts[1].trim());
        r.srcX   = Double.parseDouble(parts[2].trim());
        r.srcY   = Double.parseDouble(parts[3].trim());
        cases.add(r);
      }
    } catch (IOException e) {
      throw new RuntimeException("Failed to read cases.csv: " + e.getMessage());
    }

    sim.println("Loaded cases: " + cases.size());

    Collections.sort(cases, new Comparator<CaseRow>() {
      @Override
      public int compare(CaseRow a, CaseRow b) {
        int cU = Double.compare(a.inletU, b.inletU);
        if (cU != 0) return cU;
        int cX = Double.compare(a.srcX, b.srcX);
        if (cX != 0) return cX;
        return Double.compare(a.srcY, b.srcY);
      }
    });

    // ============================
    // 2) Get global parameters
    // ============================
    ScalarGlobalParameter inletVelocity =
      (ScalarGlobalParameter) sim.get(GlobalParameterManager.class)
        .getObject("var_inlet_velocity");

    ScalarGlobalParameter sourceX =
      (ScalarGlobalParameter) sim.get(GlobalParameterManager.class)
        .getObject("var_source_x");

    ScalarGlobalParameter sourceY =
      (ScalarGlobalParameter) sim.get(GlobalParameterManager.class)
        .getObject("var_source_y");

    Units units_ms = (Units) sim.getUnitsManager().getObject("m/s");
    Units units_m  = (Units) sim.getUnitsManager().getObject("m");

    // ============================
    // 2.5) CAD model + Part + MeshOp
    // ============================
    CadModel cadModel =
      (CadModel) sim.get(SolidModelManager.class)
        .getObject("3D-CAD Model 1");

    SolidModelPart roomPart =
      (SolidModelPart) sim.get(SimulationPartManager.class)
        .getPart("room");

    AutoMeshOperation meshOp =
      (AutoMeshOperation) sim.get(MeshOperationManager.class)
        .getObject("Automated Mesh (2D)");

    // ============================
    // 3) Get XYZ Internal Table
    // ============================
    XyzInternalTable xyzTable =
      (XyzInternalTable) sim.getTableManager()
        .getTabularObject("XYZ Internal Table");

    // ============================
    // 4) Loop over all cases
    // ============================
    //
    // Cases are sorted by inlet velocity so we can reuse the converged flow
    // field as the initial condition for different source positions under
    // the same inlet condition.
    //
    // For the first case of a new inlet velocity:
    //   - clear the full solution
    //   - run the full coupled solve
    //
    // For subsequent cases with the same inlet velocity:
    //   - keep the previously converged fields
    //   - only clear history
    //   - update source-related parameters / geometry / mesh
    //   - run again from the reused flow field
    //
    // If you later want to freeze the flow solver entirely and solve only
    // passive scalar after the first case of each U, add that STAR-specific
    // logic in the TODO block below.
    double previousU = Double.NaN;
    for (int i = 0; i < cases.size(); i++) {

      CaseRow r = cases.get(i);
      boolean firstCaseOfThisU = Double.isNaN(previousU) || !sameU(previousU, r.inletU);

      sim.println("==================================");
      sim.println("Running case " + (i+1) + "/" + cases.size()
        + " | id=" + r.caseId
        + " | u=" + r.inletU
        + " | sx=" + r.srcX
        + " | sy=" + r.srcY);
      sim.println("firstCaseOfThisU = " + firstCaseOfThisU);
      sim.println("==================================");

      // 4.1 Clear solution
      if (firstCaseOfThisU) {
        sim.println("New inlet velocity block -> clearing full solution.");
        sol.clearSolution(
          Solution.Clear.History,
          Solution.Clear.Fields,
          Solution.Clear.LagrangianDem
        );
      } else {
        sim.println("Same inlet velocity block -> reusing previous fields and clearing history only.");
        sol.clearSolution(
          Solution.Clear.History
        );
      }

      // 4.2 Set global parameters
      inletVelocity.getQuantity().setValueAndUnits(r.inletU, units_ms);
      sourceX.getQuantity().setValueAndUnits(r.srcX, units_m);
      sourceY.getQuantity().setValueAndUnits(r.srcY, units_m);

      // 4.2.1 Update CAD model
      sim.println("Updating CAD model...");
      cadModel.update();

      // 4.2.2 Update parts
      sim.println("Updating parts...");
      sim.get(SimulationPartManager.class)
        .updateParts(new ArrayList<GeometryPart>(
          Arrays.<GeometryPart>asList(roomPart)
        ));

      // 4.2.3 Remesh
      sim.println("Meshing...");
      meshOp.execute();

      // 4.3 Run simulation
      //
      // For the first case of each inlet velocity block, enable the full flow
      // + turbulence + passive scalar solve. For subsequent source positions
      // at the same inlet velocity, try to switch to scalar-only mode.
      //
      // This reflection-based toggle is intentionally defensive: if a method
      // name differs in your STAR version, the macro logs a warning instead of
      // crashing, and you can adjust the solver name/API in one place.
      if (firstCaseOfThisU) {
        sim.println("Enabling full flow + scalar solve for new inlet velocity.");
        enableFullFlowAndScalar(sim);
      } else {
        sim.println("Attempting scalar-only mode for repeated source positions at the same inlet velocity.");
        enableScalarOnlyMode(sim);
      }

      sim.println("Running simulation...");
      sim.getSimulationIterator().run();
      sim.println("✅ Simulation completed");

      // 4.4 Extract & export table
      xyzTable.extract();

      String tagU  = String.format(Locale.US, "%.4f", r.inletU);
      String tagSX = String.format(Locale.US, "%.4f", r.srcX);
      String tagSY = String.format(Locale.US, "%.4f", r.srcY);

      String outName = "case_" + r.caseId
        + "_u"  + tagU
        + "_sx" + tagSX
        + "_sy" + tagSY
        + ".csv";

      File outFile = new File(outDir, outName);
      xyzTable.export(outFile.getAbsolutePath(), ",");

      sim.println("Exported: " + outFile.getAbsolutePath());

      previousU = r.inletU;

      // optional autosave
      // if ((i + 1) % 20 == 0)
      //   sim.saveState(sessionDir + File.separator + "autosave_" + (i+1) + ".sim");
    }

    sim.println("=== All cases completed ===");
  }
}
