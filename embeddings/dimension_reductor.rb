# build: -EazybiP2
# frozen_string_literal: true



class Embeddings::DimensionReductor
  def self.pca_reduce(vector_array)
    # Define the attributes. Assume all attributes are numeric.
    attributes = Weka::Core::FastVector.new
    vector_array.first.length.times do |i|
      attributes.add_element(Weka::Core::Attribute.new("attr_#{i}"))
    end

    # Create an empty Instances object and set its attributes
    instances = Weka::Core::Instances.new(attributes: attributes)

    # Add each row to the Instances object
    vector_array.each do |row|
      instance = Weka::Core::DenseInstance.new(row.length)
      row.each_with_index do |value, index|
        instance.set_value(index, value.to_f)
      end
      instances.add_instance(instance)
    end

    pca = Java::WekaFiltersUnsupervisedAttribute::PrincipalComponents.new
    pca.set_options(Java::WekaCore::Utils.split_options("-R 1.0 -M 2"))
    pca.set_input_format(instances)

    reduced_data = pca.filter(instances)

    xy = reduced_data.map { |instance| instance.to_a }
  end

  def self.tsne_reduce(vector_array)
    # One such method is t-Distributed Stochastic Neighbor Embedding (t-SNE),
    # which is particularly good at preserving local structures
    # and has been widely used for visualizing high-dimensional data.

    # pip install scikit-learn
    # Looks like this is included in langchain install

    tempfile = Tempfile.new('serialized_input')
    tempfile.write(vector_array.to_json)
    tempfile.close
    result = `python3 tsne_script.py "#{tempfile.path}"`
    JSON.parse(result)
  end

  def self.umap_reduce(vector_array)
    # Another useful technique is Uniform Manifold Approximation and Projection (UMAP),
    # which has been shown to perform well for a variety of data types, including text embeddings.
    nil
  end

end
