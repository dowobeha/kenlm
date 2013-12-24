#include "lm/ngram_query.hh"

//modified for querying models as a preprocessing step for interpolating
//independent language models

#include "lm/model.hh"
#include "lm/word_index.hh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


int main(int argc, char *argv[]) {
  
  using namespace lm::ngram;
  using namespace lm;
  
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " lm.arpa (numSignificantDigits) < input.txt > output.scores" << std::endl;
    return -1;
  }
  
  if (argc==3) {
    int precision = atoi(argv[2]);
    std::cout.precision(precision);
  }
  
  // Open the current LM
  std::cerr << "Working on model: " << argv[1] << std::endl;
  Model model(argv[1]);
  const Vocabulary &vocab = model.GetVocabulary();
  
  // Open the list of n-grams
  std::ifstream infile(argv[1]);
  
  // Iterate through all n-grams
  for (std::string line; std::getline(std::cin, line); ) {

	  
	// For clarity, we provide a running example.
	// Assume an example input line: 
	//
	// line = "a b c d e"
	  
	  
    // Parse line into a vector of words
    std::vector<std::string> words; {
      std::stringstream stream(line);
      std::string word;
      while (stream >> word) {
        words.push_back(word);
      }
    }

	  
	// Continue the running example:
	//
	// words[0] = "a"
	// words[1] = "b"
	// words[2] = "c"
	// words[3] = "d"
	// words[4] = "e"	  
	  
	  
    // Collect word indices in a reverse-ordered array
    unsigned int num_words = words.size();
    WordIndex indices[num_words + 1];
    for (unsigned int i = 0; i < num_words; ++i) {
      indices[num_words - 1 - i] = vocab.Index(words[i]);
    }
	  
	// Continue the running example:
	//
	// num_words = 5
	//
	// indices[5] is not initialized
	// indices[4] = a
	// indices[3] = b
	// indices[2] = c
	// indices[1] = d
	// indices[0] = e
	
    
    // Output variable, which we will ignore
    State nextState;  
	  
    // Query the LM
    FullScoreReturn ret = 
      model.FullScoreForgotState(&(indices[1]), &(indices[num_words]), indices[0], nextState);
    
	// Continue the running example:
	//
	// The call to FullScoreForgotState obtains p( e | a b c d )
	  
	  
    // Report the score
    std::cout << ret.prob << std::endl;
    
  }
  
  infile.close();

}
