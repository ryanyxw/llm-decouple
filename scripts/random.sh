
numbers=0

begin="0"
end="17"

for number in $(seq 1 $numbers); do
  begin="${begin}, ${number}"
  end="$((17 - number)), ${end}"
done

echo "${begin}, ${end}"
