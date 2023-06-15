# build: -EazybiP2
# frozen_string_literal: true

class Embeddings::TextToVector

  def initialize
    # Model is currently defined in the Python script
  end

  # Initialization takes large part of runtime, super inefficient to run just for single entry.
  def self.calculate_for_array(text_array)
    serialized_input = JSON.generate(text_array)
    tempfile = Tempfile.new('serialized_input')
    tempfile.write(serialized_input)
    tempfile.close
    `python3 array_script.py "#{tempfile.path}"`.chomp
  end

end
