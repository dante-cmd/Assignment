# Assignment Scheduler using Monte Carlo Tree Search (MCTS)

This project implements an automated assignment scheduling system using Monte Carlo Tree Search (MCTS) to optimize classroom allocation for educational courses. The system assigns courses to classrooms considering capacity, scheduling conflicts, and other practical constraints across multiple periods and locations.

## Overview

The application solves the classroom assignment problem by:
- Allocating courses to appropriate classrooms based on capacity requirements
- Avoiding scheduling conflicts by checking time slot availability
- Optimizing the allocation process through MCTS algorithms
- Supporting multiple locations and time periods/frequencies

## Project Structure

```
assigment/
├── mcts_assignments.py        # Main MCTS algorithm implementation
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore file
├── project/                   # Data directory
│   ├── dim_aulas.json         # Classroom data with capacity (aforo)
│   ├── dim_frecuencia.json    # Frequency definitions (diario, sabatino, etc.)
│   ├── dim_horario.json       # Time slot definitions
│   ├── dim_periodo_franja.json # Time period and day definitions
│   ├── items.json             # Course data to be assigned
│   └── items_bimestral.json   # Already assigned courses (baseline schedule)
└── output/                    # Output directory
    └── assignments_{location}.xlsx # Output files with assignment results
```

## Data Description

- **dim_aulas.json**: Contains classroom information for each location with their names and capacities (aforo)
- **dim_frecuencia.json**: Defines scheduling frequencies including daily (Diario), weekend (Sabatino), and other patterns
- **dim_horario.json**: Maps course time ranges to individual time slots
- **dim_periodo_franja.json**: Defines time periods and corresponding time slots for different days (e.g., '1. Lun - Vie', '2. Sab')
- **items.json**: Contains data for courses that need to be assigned to classrooms
- **items_bimestral.json**: Contains data for already assigned courses (used to avoid conflicts)

## How It Works

The MCTS algorithm works in the following way:

1. **Initialization**: Creates a dataset with all necessary information and builds a RoomLog to track classroom availability
2. **MCTS Process**: Uses four phases:
   - Selection: Chooses the next node based on UCT (Upper Confidence bounds for Trees) algorithm
   - Expansion: Adds new nodes for unexplored actions
   - Simulation: Simulates random plays from the selected node
   - Backpropagation: Updates the node statistics based on simulation results
3. **Parallel Processing**: Uses multiple workers to run MCTS in parallel for better performance
4. **Assignment**: For each course, selects the best classroom assignment based on the MCTS results

## Key Classes

- **DataSet**: Loads and manages all data files from the project directory
- **RoomLog**: Tracks classroom availability and simulates assignment actions, manages scheduling constraints
- **Node**: Represents nodes in the MCTS search tree with visit counts and rewards
- **UCT**: Implements the core MCTS algorithm with UCT selection

## Usage

Run the project using:

```bash
python mcts_assignments.py --sede "Ica" --periodo_franja "1. Lun - Vie" --iter_max 5000
```

### Command Line Arguments
- `--sede`: Location/school campus (e.g. "Ica", "Chincha", "Miraflores")
- `--periodo_franja`: Time period ("1. Lun - Vie" for weekdays, "2. Sab" for weekends)
- `--iter_max`: Number of MCTS iterations per decision (default: 5000)

The output will be saved to `output/assignments_{location}.xlsx` with the assignment results for each course.

## Reward System

The algorithm assigns rewards based on:
- If classroom capacity is less than required students: `aforo - alumnos - 2`
- If capacity is sufficient but tight (within 2 of student count): `1 + (students / capacity)`
- Otherwise: `0`

This reward system prioritizes avoiding over-capacity assignments while also favoring appropriate utilization of available space.

## Locations

The dataset includes classroom assignments for multiple locations:
- Chimbote
- Chincha
- Ica
- Iquitos
- La Molina
- Lima Centro
- Lima Norte Satélite
- Lima Norte Satélite 2
- Miraflores
- Pucallpa
- San Juan de Lurigancho Satélite
- San Juan de Miraflores Satélite
- San Miguel
- Surco
- And others

## Periods and Frequencies

The system handles different scheduling periods:
- **1. Lun - Vie**: Weekday classes (Monday-Friday, Monday-Wednesday-Friday, Tuesday-Thursday)
- **2. Sab**: Weekend classes (Saturday)

With various frequencies:
- Diario (Daily)
- Sabatino (Weekend)
- Interdiario (Intermittent, e.g. L-M-W or T-TH)

## Parallel Processing

The implementation includes a parallel version of MCTS that distributes the work across multiple CPU cores to improve performance. The `parallel_UCT` function utilizes multiple processes to explore the search space concurrently, automatically using all available CPU cores on the system.

## Output

The final assignment results are exported to an Excel file (`output/assignments_{location}.xlsx`) containing:
- Original course information (code, schedule, frequency, etc.)
- Assigned classroom
- Classroom capacity
- Progress indicators during execution
- Any unassigned courses (when no suitable classroom was found)

## Algorithm Details

The MCTS implementation uses the UCT (Upper Confidence bounds for Trees) algorithm to balance exploration and exploitation during the search process. The algorithm:
- Selects actions based on UCB1 formula: `(child.w / child.visits) + c_param * sqrt(ln(parent.visits) / child.visits)`
- Expands unexplored actions when possible
- Simulates random actions to terminal state
- Backpropagates average rewards through the search tree
- Selects the action with the highest visit count as the best choice

## Dependencies

This project requires the following Python packages:
- numpy
- pandas
- pathlib
- multiprocessing
- json
- argparse
- collections


## Thanks

- https://github.com/schapman1974/tinymongo