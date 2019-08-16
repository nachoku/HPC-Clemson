   #include <mpi.h>

   main(int argc, char **argv)
   {
      int my_id, root_process, ierr, num_procs;
      MPI_Status status;

      /* Create child processes, each of which has its own variables.
       * From this point on, every process executes a separate copy
       * of this program.  Each process has a different process ID,
       * ranging from 0 to num_procs minus 1, and COPIES of all
       * variables defined in the program. No variables are shared.
       **/

      ierr = MPI_Init(&argc, &argv);
     
      /* find out MY process ID, and how many processes were started. */
      
      ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

      if( my_id == 0 ) {

         /* do some work as process 0 */
      }
      else if( my_id == 1 ) {

         /* do some work as process 1 */
      }
      else if( my_id == 2 ) {

         /* do some work as process 2 */ 
      } 
      else {

         /* do this work in any remaining processes */
      }
      /* Stop this process */

      ierr = MPI_Finalize();
   }