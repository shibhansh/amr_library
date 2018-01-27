Textual representation of AMR(Abstract Meaning Representation) can be very hard to process because of it's complex structure. This reprository provides code to simplify the process of using AMR and is aimed at increasing accessibility of AMR.

# Functionalities - 

*read_data* - Given the path to the AMR file it reads AMRs from it.
**Note**: The data in the AMR file has to be in the format specified in the gold-standard AMRs of the AMR-Bank. Future versions might include mehtods to handle other formats as well.

*amr* - Contains the AMR class, it provides following special functionalities for sentence and document AMRs - 
  - Provides translation between - (word,alignment), (alignment, node_index) etc.
  - Conversion from graphical representation of AMR to textual representation
  - Node merging with various sanity checks

*directed_graph* - Class used for the graphical representation of AMR with a useful  auxiliary functionalities

*grasshopper* - Python implementation of the grasshopper algorithm

*generate_document_graph* - Merges the sentence AMRs without coreference resolution
