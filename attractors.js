/* ============================================================
   3D trajectories for the attractor window.

   ============================================================ */

window.TRAJECTORIES = [
  // {
  //   name: "my-run",
  //   points: [
  //     [0.1, 0.0, 0.0],
  //     [0.12, 0.03, 0.01],
  //     ...
  //   ]
  // },
];

/* ------------------------------------------------------------
   Generating this file from Julia:

     using JSON3
     traj = [ [x[i], y[i], z[i]] for i in eachindex(x) ]   # your solution
     open("attractors.js", "w") do io
         write(io, "window.TRAJECTORIES = ")
         JSON3.write(io, [(name = "my-run", points = traj)])
         write(io, ";")
     end

   ...or from Python:

     import json
     data = [{"name": "my-run", "points": pts}]   # pts = list of [x,y,z]
     open("attractors.js", "w").write(
         "window.TRAJECTORIES = " + json.dumps(data) + ";")
   ------------------------------------------------------------ */
