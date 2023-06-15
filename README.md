# Calculate embeddings for imported issue descriptions.

# 1) Setup DB and table. This all was done in Postgres, dwh_22 schema.

CREATE EXTENSION vector;

alter table dwh_22.jira_issues
add column description_embedding vector(384);


# 2) Manual data processing after import. This was originally all done through Rails console. A better way could be to do this all outside of eazyBI and just import the reduced dimensions as additional data.

issue_data = Dwh.select_all "select id, customfield_description_for_embedding, issue_summary from dwh_22.jira_issues where id > 0"

# Not every issue has a description, so we use summary as well
texts = issue_data.map do |issue|
  text = (issue["issue_summary"]||"").split(" ")[1..-1].join(" ") + ". "
  if description = issue["customfield_description_for_embedding"]
    text += description.gsub("\n", " ")
  end
  text
end

# Calculate embeddings. This takes some time.
vectors = Embeddings::TextToVector.calculate_for_array texts

# vectors is a string that contains JSON array of vectors
vector_array = JSON.parse vectors

# vector_array is an array of vectors
vector_array.each_with_index do |vector, index|
  Dwh.update <<~SQL
    UPDATE dwh_22.jira_issues
    SET description_embedding = '#{vector}'
    WHERE id = #{issue_data[index]["id"]}
  SQL
end




# Reduce dimensionality of embeddings.
issue_data = Dwh.select_all "select id, description_embedding from dwh_22.jira_issues where id > 0"

data = issue_data.map do |issue|
  JSON.parse(issue["description_embedding"])
end

# There exist multiple methods for dimensionality reduction. Here is implementation for two of them.
# PCA - Principal Component Analysis
pca_reduced_data = Embeddings::DimensionReductor.pca_reduce(data)

xy = pca_reduced_data.map { |instance| instance.to_a.map { |v| v * 10000 } }

# Put it back into database in the right fields.
# The fields are defined as custom fields in import options. Imported as properties.
xy.each_with_index do |vector, index|
  Dwh.update <<~SQL
    UPDATE dwh_22.jira_issues
    SET customfield_pca_x = #{vector[0]}, customfield_pca_y = #{vector[1]}
    WHERE id = #{issue_data[index]["id"]}
  SQL
end


# TSNE - t-Distributed Stochastic Neighbor Embedding
tsne_reduced_data = Embeddings::DimensionReductor.tsne_reduce(data)

xy = tsne_reduced_data.map { |instance| instance.to_a.map { |v| v * 10000 } }

# put it back into database
xy.each_with_index do |vector, index|
  Dwh.update <<~SQL
    UPDATE dwh_22.jira_issues
    SET customfield_tsne_x = #{vector[0]}, customfield_tsne_y = #{vector[1]}
    WHERE id = #{issue_data[index]["id"]}
  SQL
end
